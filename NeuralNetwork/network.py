from preprocessing import pullNormalisedDataFromCSV

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

import tensorflow as tf
import pandas as pd
import numpy as np
import os



def splitData(data, seqL, step):

    df = data

    # X becomes features, y becomes targets.
    # although X becomes features, the target columns are not dropped so they
    #   can be accessed during sequence creation and dropped then
    X = df
    y = df[["P1_left", "P1_right", "P1_attack"]]

    # splits the data 70/30 into a feature and target training and testing
    # random state = 17 ensures the split is always the same
    # note: y train, val, and raw are never actually accessed since these 
    #   targets are created in the createSequences function, they're just here
    #   since 
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(X, y, test_size=0.3, random_state=17)

    # split the what was 30 percent remaining into 50/50
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(X_temp, y_temp, test_size=0.5, random_state=17)

    # create sequences for training, validation, and test sets
    X_train_seq, y_train_seq = createSequences(X_train_raw, seqL, step)
    X_val_seq, y_val_seq = createSequences(X_val_raw, seqL, step)
    X_test_seq, y_test_seq = createSequences(X_test_raw, seqL, step)

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq


def createSequences(df, seqL, step):
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
            sequence = roundData.iloc[i:i+seqL].drop(columns=targetColumns)
            target = roundData.iloc[i:i+seqL][targetColumns]

            # this pulls the values from the iloc methods.
            sequences.append(sequence.values)
            targets.append(target.values)
    
    return np.array(sequences), np.array(targets)


def buildModel(inputShape):
    
    # sequential model, 1 input, 1 output, 2 hidden layers
    # LSTM layer with 64 units, dense with 32, and dense with 3
    # last dense layer is output layer w sigmoid function
    model = Sequential()
    model.add(Input(shape=inputShape))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    # compiling model with adam, binary cross-entropy, and accuracy metric
    model.compile(optimizer=Adam(), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model


def main():

    # pulls data, adjusts hyperparams such as sequence length and overlap.
    df = pullNormalisedDataFromCSV()
    sequenceLength = 20
    sequenceOverlap = 1
    epochs = 10
    batchSize = 64

    # calls for data splitting
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq = splitData(
        data=df,
        seqL=sequenceLength,
        step=sequenceOverlap)
    
    # this takes the shape of the data gathered, so frames per sequence and
    #   features per timestep
    # shape[0] is equal to total sequences
    inputShape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = buildModel(inputShape=inputShape)
    checkpoint = ModelCheckpoint("peak.keras", 
                                 save_best_only=True, 
                                 monitor="val_loss", 
                                 mode="min", 
                                 verbose=1)

    # Now, train the model with the checkpoint callback
    history = model.fit(X_train_seq, 
                        y_train_seq, 
                        epochs=epochs, 
                        batch_size=batchSize, 
                        validation_data=(X_val_seq, y_val_seq), 
                        callbacks=[checkpoint])


if __name__ == "__main__":
    main()