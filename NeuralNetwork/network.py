from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dropout,  Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard # type: ignore
from tensorflow.keras.metrics import Metric # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

from preProcessing import DataPreprocessor

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras, os, datetime, json

# run this to open tensorboard
# python -m tensorboard.main --logdir="C:\Users\bahaf\Documents\Final Year Project\footsies_final_year_project\NeuralNetwork\experimentLogs"

@keras.saving.register_keras_serializable()
def weightedBinaryCrossentropy(classWeights):
    """
    returns a custom loss function with class specific weights.
    """
    classWeights = tf.constant(classWeights, dtype=tf.float32)

    @keras.saving.register_keras_serializable()
    def loss_fn(y_true, y_pred):
        # clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # apply the binary cross entropy formula with the weights
        bce = -(classWeights * y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))

        return tf.reduce_mean(bce)  # average over batch and classes

    return loss_fn

@keras.saving.register_keras_serializable()
class F1Score(Metric):
    '''
    custom class inhereting the keras Metric class
    calculated the F score of a model
    '''
    def __init__(self, num_classes=3, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # predictions to binary w threshold 0.5
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # flatten to 2d
        if len(y_pred.shape) == 3:
            y_pred = tf.reshape(y_pred, [-1, self.num_classes])
            y_true = tf.reshape(y_true, [-1, self.num_classes])

        # true and false positives, and false negatives
        tp = tf.reduce_sum(y_pred * y_true)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

    # the f score formula
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


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


def retrieveClassWeights(df):
    # defining weights for the custom weightedBinaryCrossentropy function
    targets = df[["P1_left", "P1_right", "P1_attack"]]
    classWeights = []
    for col in targets.columns:
        n_pos = targets[col].sum()
        n_total = len(targets[col])
        n_neg = n_total - n_pos
        pos_weight = n_neg / n_pos if n_pos != 0 else 1.0
        classWeights.append(round(pos_weight, 2))
    
    return classWeights

def buildModel(inputShape, lstmSize, dropoutRate, denseSize, classWeights):
    '''
    builds a model with passed arguments defining the model architecture
    '''
    
    # sequential model, 1 input, 1 output, 2 hidden layers
    # LSTM layer with 64 units, 0.3 dropout layer, dense with 32, and dense with 3
    # last dense layer is output layer w sigmoid function
    model = Sequential()
    model.add(Input(shape=inputShape))
    model.add(LSTM(lstmSize, return_sequences=True))
    model.add(Dropout(dropoutRate))
    model.add(Dense(denseSize, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    # compiling model with adam, binary cross-entropy, and accuracy metric
    model.compile(optimizer=Adam(), 
                    loss=weightedBinaryCrossentropy(classWeights), 
                    metrics=['accuracy', F1Score()])

    return model

def defineCallbacks(logPath, modelPath):
    '''
    defines all callbacks, keep them together so its neater
    '''

    checkpointCB = ModelCheckpoint(modelPath, 
                                    save_best_only=True, 
                                    monitor="val_f1_score", 
                                    mode="max", 
                                    verbose=1)
    
    earlystopCB = EarlyStopping(monitor='val_f1_score',
                                mode='max',
                                patience=5,
                                restore_best_weights=True,
                                verbose=1)

    tensorboardCB = TensorBoard(log_dir=logPath, 
                                histogram_freq = 1)
    
    return [checkpointCB, earlystopCB, tensorboardCB]


def permutationImportance(model, X, y, metric_class):
    """
    model: trained keras model
    X: input features (num_samples, seq_len, features)
    y: true targets (num_samples, seq_len, num_classes)
    metric_class: a keras Metric class
    """
    # Instantiate the metric
    metric = metric_class()
    
    # Get base score
    y_pred = model.predict(X)
    metric.update_state(y, y_pred)
    base_score = metric.result().numpy()
    metric.reset_states()

    importances = []
    feature_count = X.shape[2]

    for i in range(feature_count):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, :, i])  # shuffle feature i across all samples

        y_pred_perm = model.predict(X_permuted)
        metric.update_state(y, y_pred_perm)
        permuted_score = metric.result().numpy()
        metric.reset_states()

        importance = base_score - permuted_score
        importances.append(importance)

    return np.array(importances)


def standardModelTrainTest():
    '''
    the standard model test, using the default configuration of hyperparameters
    '''

    try:
        myNormaliser = DataPreprocessor()
        df = myNormaliser.pullDataset()
    except Exception as e:
        print("PreProcessor could not be loaded.")
        print(e)

    epochs = 50
    logPath = os.path.join(os.path.dirname(__file__),f"experimentLogs\\{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}")
    classWeights, sequenceLength, sequenceStep, batchSize = retrieveClassWeights(df), 20, 1, 32
    lstmSize, dropoutRate, denseSize = 64, 0.5, 32


    # splits data and creates sequences
    try:
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = splitData(
            df=df,
            seqL=sequenceLength,
            step=sequenceStep)
    except Exception as e:
        print("network.splitData() could not be run.")
        print(e)
    

    # defines input shape from created sequences
    try:
        inputShape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = buildModel(inputShape=inputShape, 
                           lstmSize=lstmSize, 
                           dropoutRate=dropoutRate, 
                           denseSize=denseSize,
                           classWeights=classWeights)
    except Exception as e:
        print(f"Error creating model: {e}")


    # train the model with the checkpoint callback
    history = model.fit(X_train_seq, 
                        y_train_seq, 
                        epochs=epochs, 
                        batch_size=batchSize, 
                        validation_data=(X_val_seq, y_val_seq), 
                        callbacks=defineCallbacks(logPath, "FootsiesNeuralNetwork.keras"))
    

    model.evaluate(X_test_seq, y_test_seq, batch_size=batchSize)

    importances = permutationImportance(model, X_val_seq, y_val_seq, metric_class=F1Score)
    featureLabels = (df.drop(columns=["P1_attack", "P1_left", "P1_right", "round_ID", "frame_number"])).columns.tolist()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances)
    plt.yticks(ticks=range(len(importances)), labels=featureLabels)
    plt.xlabel("Importance (ΔF1 score)")
    plt.title("Feature Importance (Permutation)")
    plt.tight_layout()
    plt.show()

    return history

def batchHyperparamTest():
    '''
    performs a batch test of all configuration files to find the optimal
        hyperparameter layout
    '''

    try:
        myNormaliser = DataPreprocessor()
        df = myNormaliser.pullDataset()
    except Exception as e:
        print("PreProcessor could not be loaded.")
        print(e)

    # splits data and creates sequences
    try:
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = splitData(
            df=df,
            seqL=20,
            step=1)
    except Exception as e:
        print("network.splitData() could not be run.")
        print(e)

    epochs = 50

    configDir = os.path.join(os.path.dirname(__file__),"experimentConfigs\\hyperparameters")
    configFiles = [f for f in os.listdir(configDir) if f.endswith(".json")]

    for config in configFiles:
        experimentName = config[:config.index(".json")]

        experimentPath = os.path.join(
            os.path.dirname(__file__),
            f"experimentConfigs\\hyperparameters\\{experimentName}.json")

        experimentModel = os.path.join(
            os.path.dirname(__file__),
            f"experimentModels\\{experimentName}.keras")
        
        logPath = os.path.join(
            os.path.dirname(__file__),
            f"experimentLogs\\{experimentName}\\{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}")
        
        try:
            with open(experimentPath) as rawConfig:
                config = json.load(rawConfig)

                batchSize = config["training"]["batch_size"]
                sequenceLength = config["training"]["sequence_length"]
                sequenceStep = config["training"]["step"]
                lstmSize = config["model"]["LSTM_unit_size"]
                dropoutRate = config["model"]["dropout_rate"]
                denseSize = config["model"]["dense_unit_size"]

        except Exception as e:
            print(f"Unable to read config file: {e}")

        if (sequenceLength != 20) or (sequenceStep != 1):
            print("Regenerating training sequences.")
            try:
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = splitData(
                    df=df,
                    seqL=sequenceLength,
                    step=sequenceStep)
            except Exception as e:
                print(f"network.splitData() could not be run: {e}")

        classWeights = retrieveClassWeights(df)

        print(f"Running experiment at path: {experimentPath}", end="\n")

        # defines input shape from created sequences
        try:
            inputShape = (X_train_seq.shape[1], X_train_seq.shape[2])
            model = buildModel(inputShape=inputShape, 
                            lstmSize=lstmSize, 
                            dropoutRate=dropoutRate, 
                            denseSize=denseSize,
                            classWeights=classWeights)
        except Exception as e:
            print(f"Error creating model: {e}")


        # train the model with the checkpoint callback
        print(model.summary())
        print(f"Batch Size: {batchSize}\nSequence Length: {sequenceLength}\nSequence Step: {sequenceStep}\n")
        model.fit(X_train_seq, 
                  y_train_seq, 
                  epochs=epochs, 
                  batch_size=batchSize, 
                  validation_data=(X_val_seq, y_val_seq), 
                  callbacks=defineCallbacks(logPath, experimentModel))
        
        model.evaluate(X_test_seq, y_test_seq, batch_size=batchSize)


def main():
    
    history = standardModelTrainTest()

    def plotHistory(history):
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.show()

    plotHistory(history)


if __name__ == "__main__":
    main()