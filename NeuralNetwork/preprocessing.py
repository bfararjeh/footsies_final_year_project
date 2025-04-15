import pandas as pd
import numpy as np
import re, os
'''
dummy line for testing

49: P1_INFO:currentInput(1)position(-1.16, 0.00)velocity_x(0)isDead(False)vitalHealth(1)guardHealth(3)currentActionID(105)currentActionFrame(10)currentActionFrameCount(21)isAlwaysCancelable(False)currentActionHitCount(1)currentHitStunFrame(0)isInHitStun(False)isAlwaysCancelable(False)P2_INFO:currentInput(2)position(0.93, 0.00)velocity_x(-0.5)isDead(False)vitalHealth(1)guardHealth(2)currentActionID(305)currentActionFrame(7)currentActionFrameCount(15)isAlwaysCancelable(False)currentActionHitCount(0)currentHitStunFrame(0)isInHitStun(False)isAlwaysCancelable(False)
'''

def parse_line(lineToParse):

    frame_match = re.match(r"(\d+): ", lineToParse)
    frame_number = int(frame_match.group(1))

    P1_data = lineToParse[lineToParse.index("P1_INFO:"):
                          lineToParse.index("P2_INFO:")]
    P2_data = lineToParse[lineToParse.index("P2_INFO:"):]

    pattern = r"currentInput\((\d+)\)" + \
        r"position\(([-\d.]+), 0.00\)" + \
        r"velocity_x\((\d+)\)" + \
        r"isDead\((True|False)\)" + \
        r"vitalHealth\((\d+)\)" + \
        r"guardHealth\((\d+)\)" + \
        r"currentActionID\((\d+)\)" + \
        r"currentActionFrame\((\d+)\)" + \
        r"currentActionFrameCount\((\d+)\)" + \
        r"isAlwaysCancelable\((True|False)\)" + \
        r"currentActionHitCount\((\d+)\)" + \
        r"currentHitStunFrame\((\d+)\)" + \
        r"isInHitStun\((True|False)\)"

    P1matches = re.search(pattern, P1_data)
    P2matches = re.search(pattern, P2_data)

    return {

    }

def convert_dataframe(rawdata):
    pass

def parse_dataset(path):

    '''bit messy, but this grabs the path of the training data with easy 
    changing of dataset path'''
    dataset_name = r"TrainingData\DATASET#2-NEW_AI"
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        dataset_name)

    for trainingFile in os.listdir(dataset_path):
        pass


def main():
    pass

if __name__ == "__main__":
    main()