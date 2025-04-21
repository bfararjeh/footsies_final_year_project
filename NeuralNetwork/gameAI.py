import numpy as np
import pandas as pd
import tensorflow as tf

from normalisation import Normaliser
from dataParsing import parse_line

from tensorflow.keras.models import load_model # type: ignore

class FootsiesPredictor():
    def __init__(self, modelPath, sequenceLength, features):
        self.model = load_model(modelPath)
        self.seqL = sequenceLength
        self.features = features
        self.buffer = [] 
        self.myNormaliser = Normaliser()


    def prepareData(self, rawLine):
        _, parsedLine = parse_line(rawLine)
        normalisedFrame = self.myNormaliser.normaliseLine(parsedLine)
        normalisedFrame = normalisedFrame.drop(["round_ID", "frame_number"], errors="ignore")
        return normalisedFrame


    def addFrame(self, normalisedFrame):
        self.buffer.append(normalisedFrame)

        if len(self.buffer) > self.seqL:
            self.buffer.pop(0)


    def ready(self):
        return len(self.buffer) == self.seqL


    def predict(self):

        if self.ready() == False:
            finalOutput = 0
        
        else:
            currentSequence = np.array(self.buffer, dtype=np.float32).reshape((1, self.seqL, self.features))
            prediction = self.model(currentSequence)
            threshold = 0.01
            binOutput = [
                1 if value >= threshold else 0 for value in prediction.numpy()[0][-1].tolist()]
            binString = ''.join(str(bit) for bit in binOutput[::-1])

            finalOutput = int(binString, 2)
            
        return finalOutput