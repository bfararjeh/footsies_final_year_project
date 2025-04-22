import numpy as np
import pandas as pd
import tensorflow as tf

from preProcessing import DataPreprocessor

from tensorflow.keras.models import load_model # type: ignore

class FootsiesPredictor():
    def __init__(self, modelPath, sequenceLength, features):
        self.model = load_model(modelPath)
        self.seqL = sequenceLength
        self.features = features
        self.buffer = [] 
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

        if len(self.buffer) == 1:
            self.buffer = [normalisedFrame] * self.seqL

        if len(self.buffer) > self.seqL:
            self.buffer.pop(0)

    def ready(self):
        return len(self.buffer) == self.seqL

    def predict(self):

        if self.ready() == False:
            finalOutput = 0
        
        else:
            currentSequence = np.array(self.buffer, dtype=np.float32).reshape(
                (1, self.seqL, self.features))
            prediction = self.model(currentSequence)[0, -1]
            threshold = 0.5
            binOutput = [
                1 if value >= threshold else 0 for value in prediction.numpy().tolist()]
            binString = ''.join(str(bit) for bit in binOutput[::-1])

            finalOutput = int(binString, 2)
            print(finalOutput)
            
        return finalOutput