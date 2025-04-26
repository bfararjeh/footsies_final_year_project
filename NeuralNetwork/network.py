from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dropout,  Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

from preProcessing import DataPreprocessor

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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

def buildModel(inputShape):
    
    # sequential model, 1 input, 1 output, 2 hidden layers
    # LSTM layer with 64 units, 0.3 dropout layer, dense with 32, and dense with 3
    # last dense layer is output layer w sigmoid function
    model = Sequential()
    model.add(Input(shape=inputShape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    # compiling model with adam, binary cross-entropy, and accuracy metric
    model.compile(optimizer=Adam(), 
                    loss='binary_crossentropy', 
                    metrics=["accuracy"])

    return model

def loadBestModel():
    loadedModel = load_model("peak.keras")
    return loadedModel

def plotHistory(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

def main():
    '''
    this function is the MAIN model building function. this is the final model,
        results from experiments should be used to change this function
        specifically.
    '''

    # pulls data, adjusts hyperparams such as sequence length and overlap.
    try:
        myNormaliser = DataPreprocessor()
        df = myNormaliser.pullDataset()
    except Exception as e:
        print("PreProcessor could not be loaded.")
        print(e)

    sequenceLength = 20
    sequenceOverlap = 1
    epochs = 30
    batchSize = 32

    # calls for data splitting
    try:
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = splitData(
            df=df,
            seqL=sequenceLength,
            step=sequenceOverlap)
    except Exception as e:
        print("network.splitData() could not be run.")
        print(e)


    # this takes the shape of the data gathered, so frames per sequence and
    #   features per timestep
    # shape[0] is equal to total sequences
    try:
        inputShape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = buildModel(inputShape=inputShape)
        checkpoint = ModelCheckpoint("FootsiesNeuralNetwork.keras", 
                                        save_best_only=True, 
                                        monitor="val_loss", 
                                        mode="min", 
                                        verbose=1)
        earlyStop = EarlyStopping(monitor='val_loss',
                                patience=6,
                                restore_best_weights=True,
                                verbose=1)
    except Exception as e:
        print(f"Error creating model: {e}")

    print(f"Input shape: {inputShape}")
    print(f"X_train_seq shape: {X_train_seq.shape}")
    assert inputShape == X_train_seq.shape[1:], "Input shape doesn't match training data"

    # train the model with the checkpoint callback
    history = model.fit(X_train_seq, 
                        y_train_seq, 
                        epochs=epochs, 
                        batch_size=batchSize, 
                        validation_data=(X_val_seq, y_val_seq), 
                        callbacks=[checkpoint, earlyStop])

    trainingBatchesEst = int(np.ceil(len(X_train_seq) / batchSize))
    print(f"Expected number of batches per epoch: {trainingBatchesEst}")
    plotHistory(history)

    # loads then evaluates the best model
    model = load_model("FootsiesNeuralNetwork.keras")
    model.evaluate(X_test_seq, y_test_seq, batch_size=batchSize)


if __name__ == "__main__":
    main()