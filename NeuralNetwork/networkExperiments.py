from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dropout,  Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping # type: ignore

from preProcessing import DataPreprocessor

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os, json, datetime

# run this command to view tensorboard logs
# python -m tensorboard.main --logdir="C:\Users\bahaf\Documents\Final Year Project\footsies_final_year_project\NeuralNetwork\experimentLogs"

def splitData(df, seqL, step):
    '''
    splits data into multiple sequences to be fed into the network during 
        training
    method takes params data, sequence length, and sequence step
    step is used to determine overlap of sequence windows
    '''

    # X becomes features, y becomes targets.
    # although X becomes features, the target columns are not dropped so they
    #   can be accessed during sequence creation and dropped then
    X = df
    y = df[["P1_left", "P1_right", "P1_attack"]]


    # splits dataframes to unique rounds, to ensure rounds stay together
    # splits the data 70/30 into a feature and target training and testing
    # random state = 17 ensures the split is always the same
    unique_round_ids = df['round_ID'].unique()
    train_ids, temp_ids = train_test_split(unique_round_ids, test_size=0.3, random_state=17)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=17)

    # alligns target and features to the split round IDs
    X_train_raw = X[X['round_ID'].isin(train_ids)]
    y_train_raw = y[X['round_ID'].isin(train_ids)]

    X_val_raw = X[X['round_ID'].isin(val_ids)]
    y_val_raw = y[X['round_ID'].isin(val_ids)]

    X_test_raw = X[X['round_ID'].isin(test_ids)]
    y_test_raw = y[X['round_ID'].isin(test_ids)]

    # create sequences for training, validation, and test sets
    X_train_seq, y_train_seq = createSequences(X_train_raw, seqL, step)
    X_val_seq, y_val_seq = createSequences(X_val_raw, seqL, step)
    X_test_seq, y_test_seq = createSequences(X_test_raw, seqL, step)

    assert X_train_seq.shape[0] > 0, "Training set is empty!"
    assert X_val_seq.shape[0] > 0, "Validation set is empty!"
    assert X_test_seq.shape[0] > 0, "Test set is empty!"

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq

def createSequences(df, seqL, step):
    '''
    iterates through the data and returns two created arrays of sequences,
        one for features and one for targets
    '''
    sequences = []
    targets = []
    targetColumns = [
        "P1_left",
        "P1_right",
        "P1_attack",]

    # the _ is for roundID, while not accessed, its here to make sure sequences
    #   dont bleed into other rounds
    for _, roundData in df.groupby("round_ID"):

        # this skips rounds that are too short, which is impossible rn, but
        #   if i ever make seqL larger then could be relevant
        if len(roundData) < seqL:
            continue 
        
        # sorts the round by frame number
        roundData = roundData.sort_values(by="frame_number")

        # loops the round data in range length round data to sequence length, 
        #   + 1 to account for range exclusiveness
        # the step here is 1, this creates a sliding window 
        for i in range(0, len(roundData) - seqL + 1, step):

            # this grabs all the data apart from the targets, or only the
            #   targets
            # .iloc simply grabs data at an index or index range, specifying
            #   the column to grab from
            sequence = roundData.iloc[i:i+seqL].drop(columns=targetColumns + ["round_ID", "frame_number"])
            target = roundData.iloc[i:i+seqL][targetColumns]

            # this pulls the values from the iloc methods.
            sequences.append(sequence.values)
            targets.append(target.values)
    
    # small two lines of code to ensure sequences created correctly
    print(f"Created {len(sequences)} sequences with shape {sequences[0].shape if sequences else 'N/A'}")
    assert len(sequences) == len(targets), "Mismatch between sequences and targets"

    return np.array(sequences), np.array(targets)


def lenient_accuracy(y_true, y_pred, window=2):
    """
    find lenient accuracy
    y_true: (batch_size, seq_length, num_classes)
    y_pred: (batch_size, seq_length, num_classes)
    """
    # Round predictions to nearest 0 or 1
    y_pred_rounded = tf.round(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    
    # Initialize matches
    batch_size = tf.shape(y_true)[0]
    seq_len = tf.shape(y_true)[1]

    correct = 0.0
    total = 0.0

    for offset in range(-window, window + 1):
        # Shift y_true
        shifted_y_true = tf.roll(y_true, shift=offset, axis=1)

        # Compare predictions to shifted ground truth
        matches = tf.reduce_all(tf.equal(y_pred_rounded, shifted_y_true), axis=-1)  # (batch_size, seq_len)

        # Ignore invalid shifts (over borders)
        if offset < 0:
            matches = matches[:, :-offset]
        elif offset > 0:
            matches = matches[:, offset:]

        correct += tf.reduce_sum(tf.cast(matches, tf.float32))
        total += tf.cast(tf.size(matches), tf.float32)

    return correct / total

class LenientAccuracy(tf.keras.metrics.Metric):
    def __init__(self, window=2, name="lenient_accuracy", **kwargs):
        super(LenientAccuracy, self).__init__(name=name, **kwargs)
        self.window = window
        self.accuracy = self.add_weight(name="accuracy", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        acc = lenient_accuracy(y_true, y_pred, window=self.window)
        self.accuracy.assign_add(acc)
        self.count.assign_add(1.0)

    def result(self):
        return self.accuracy / self.count

    def reset_states(self):
        self.accuracy.assign(0.0)
        self.count.assign(0.0)


def runExperiment(experimentName):
    '''
    this function is for experimental model building, used for testing different
        configurations.
    this and the main method are messy, but they dont need to be clean. would
        rather one large function than many smaller ones in this specific
        instance
    '''

    # defines path of config file, model, and log
    experimentPath = os.path.join(
        os.path.dirname(__file__),
        f"experimentConfigs\\{experimentName}.json")

    experimentModel = os.path.join(
        os.path.dirname(__file__),
        f"experimentModels\\{experimentName}.keras")
    
    logPath = os.path.join(
        os.path.dirname(__file__),
        f"experimentLogs\\{experimentName}\\{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}")
    
    print(f"Running experiment at path: {experimentPath}", end="\n")

    # reads experiment config
    try:
        with open(os.path.join("experimentConfigs/", experimentPath)) as config:
            config = json.load(config)

    except Exception as e:
        print(f"Unable to read config file: {e}")

    # pulls data, adjusts hyperparams such as sequence length and overlap.
    try:
        myNormaliser = DataPreprocessor()
        df = myNormaliser.pullDataset()
    except Exception as e:
        print("PreProcessor could not be loaded.")
        print(e)

    # calls for data splitting
    try:
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = splitData(
            df=df,
            seqL=config["training"]["sequence_length"],
            step=config["training"]["step"])
    except Exception as e:
        print(f"Data could not be split: {e}")

    # this takes the shape of the data gathered, so frames per sequence and
    #   features per timestep
    # shape[0] is equal to total sequences
    try:
        inputShape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = Sequential()
        model.add(Input(shape=inputShape))
        model.add(LSTM(config["model"]["LSTM_unit_size"], return_sequences=True))
        model.add(Dropout(config["model"]["dropout_rate"]))
        model.add(Dense(config["model"]["dense_unit_size"], activation='relu', dtype="float32"))
        model.add(Dense(3, activation='sigmoid'))

        newAdam = Adam(learning_rate=config["model"]["learning_rate"])

        # compiling model with adam, binary cross-entropy, and accuracy metric
        model.compile(optimizer=newAdam, 
                        loss='binary_crossentropy', 
                        metrics=['accuracy', LenientAccuracy(window=2)])
    except Exception as e:
        print(f"Error creating model: {e}")

    tensorboard = TensorBoard(log_dir=logPath, histogram_freq = 1)
    checkpoint = ModelCheckpoint(experimentModel, 
                                    save_best_only=True, 
                                    monitor="val_loss", 
                                    mode="min", 
                                    verbose=1)
    earlyStop = EarlyStopping(monitor='val_loss',
                              patience=config["model"]["early_stopping_patience"],
                              restore_best_weights=True,
                              verbose=1)

    # train the model with the tensorboard callback
    model.fit(X_train_seq, 
              y_train_seq, 
              epochs=config["training"]["epochs"], 
              batch_size=config["training"]["batch_size"], 
              validation_data=(X_val_seq, y_val_seq), 
              callbacks=[tensorboard, checkpoint, earlyStop])






if __name__ == "__main__":

    configDir = os.path.join(os.path.dirname(__file__),"experimentConfigs")
    configFiles = [f for f in os.listdir(configDir) if f.endswith(".json")]

    for config in configFiles:
        currentConfig = config[:config.index(".json")]
        runExperiment(currentConfig)