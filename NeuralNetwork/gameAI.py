import numpy as np
import pandas as pd
import tensorflow as tf

from preProcessing import DataPreprocessor
from network import F1Score, weightedBinaryCrossentropy
from tensorflow.keras.models import load_model # type: ignore


class FootsiesPredictor():
    '''
    This class is the entirety of the game AI. It handles normalisation of
        live inputs, sequence generation and prediction.
    An instance of this is created in "server.py", and messages are passed to
        it every frmae.

    Its methods:
        prepareData: normalises a single line of data and then adds it to the
            current sequence buffer
        addFrame: called by the above method, adds a frame to the buffer, while
            also padding the buffer and ensuring it remains <=20
        predict: predicts the next input using the current buffer and the model
            specified in the argument "modelPath"
    
    Argument type: path, int, int, int
    '''

    def __init__(self, modelPath, sequenceLength, features, predictIntervals):
        self.model = load_model(modelPath, custom_objects={
            "F1Score": F1Score,
            "loss_fn": weightedBinaryCrossentropy([np.float64(2.92), np.float64(2.16), np.float64(5.53)])}
        )
        self.seqL = sequenceLength
        self.features = features
        self.predInterval = predictIntervals

        # buffer and counter for managing prediction rates and sequences
        self.internalCounter = 1
        self.lastPrediction = 0
        self.buffer = [] 

        # instance of preprocessor for normalisation
        self.myPreprocessor = DataPreprocessor()

    def prepareData(self, rawLine):
        normalisedLine = self.myPreprocessor.normaliseLiveInput(rawLine)

        # ensures all columns present (mainly for one hot encoded cols)
        for col in self.myPreprocessor.normalisedColumnList:
            if col not in normalisedLine.columns:
                normalisedLine[col] = 0

        # this reorders data to the same order as the training set
        normalisedLine = normalisedLine[self.myPreprocessor.normalisedColumnList]
        
        # drop targets + cols dropped in sequencing
        normalisedLine.drop(columns=[
            "P1_attack",
            "P1_right",
            "P1_left",
            "round_ID",
            "frame_number"], inplace=True, errors="ignore")
        
        self.addFrame(normalisedLine)

    def addFrame(self, normalisedFrame):
        self.buffer.append(normalisedFrame)
        self.internalCounter += 1

        if len(self.buffer) == 1:
            self.buffer = [normalisedFrame] * self.seqL

        if len(self.buffer) > self.seqL:
            self.buffer.pop(0)

    def predict(self):
        
        # runs when buffer length = sequence length and the prediction interval is met
        if (len(self.buffer) == self.seqL) & (self.internalCounter % self.predInterval == 0):

            # rearrange buffer
            currentSequence = np.array(self.buffer, dtype=np.float32).reshape(
                (1, self.seqL, self.features))
            
            # create prediction with model
            # prediction order is [left, right, attack]
            prediction = self.model(currentSequence)[0, -1]

            threshold = 0.4         # input threshold

            # converts list of floats into binary, then into int
            binOutput = [
                1 if value >= threshold else 0 for value in prediction.numpy().tolist()]
            binString = ''.join(str(bit) for bit in binOutput[::-1])
            finalOutput = int(binString, 2)

            self.lastPrediction = finalOutput
        
        # lastPrediction ensures something is always returned, even if its old
        return self.lastPrediction