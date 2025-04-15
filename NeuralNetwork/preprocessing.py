from dataParsing import pullDataFromCSV
import pandas as pd
import numpy as np
import re, os

'''
every variable to be normalised, taken from pattern variable in dataparsing
        r"currentInput\((\d+)\)" + \
        r"position\(([-\d.]+), 0.00\)" + \
        r"velocity_x\(([-\d.]+)\)" + \
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
'''

def normalise(df):

    print(df)
    
    df = pd.get_dummies(df,
                        columns=["P1_currentActionID", "P2_currentActionID"],
                        prefix=["P1_moveID", "P2_moveID"])
    
    '''
    magic bit shit. basically, the "currentInput" is a bitwise mask that takes
        the binary values of button presses (the three buttons left right and
        attack) and then masks them into a single integer.
    this code essentially decomposes the bitmask, creating three new columns
        in the dataframe per player for each attack
    im not 100% how the bit stuff works, its kinda losing me, but regardless it
        works
    below is the bit mapping the game uses

        L		    R		    A           C
        0			0			0		    =0
        1			0			0		    =1
        0			1			0		    =2
        1			1			0		    =3
        0			0			1		    =4
        1			0			1		    =5
        0			1			1 		    =6
        1			1			1		    =7

    where L is left, R is right, A is attack, and C is currentInput
    '''
    df['P1_left']   = np.bitwise_and(
        np.right_shift(df["P1_currentInput"].astype(int), 2), 1)  # bit 2
    df['P1_right']  = np.bitwise_and(
        np.right_shift(df["P1_currentInput"].astype(int), 1), 1)  # bit 1
    df['P1_attack'] = np.bitwise_and(
        df["P1_currentInput"].astype(int), 1)                     # bit 0
    df.drop(columns='P1_currentInput', inplace=True)    # drop the old column

    df['P2_left']   = np.bitwise_and(
        np.right_shift(df["P2_currentInput"].astype(int), 2), 1)  # bit 2
    df['P2_right']  = np.bitwise_and(
        np.right_shift(df["P2_currentInput"].astype(int), 1), 1)  # bit 1
    df['P2_attack'] = np.bitwise_and(
        df["P2_currentInput"].astype(int), 1)                     # bit 0
    df.drop(columns='P2_currentInput', inplace=True)    # drop the old column

    
    
    
    
    
    
    
    
    
    
    print(df)


def main():
    normalise(pullDataFromCSV())


if __name__ == "__main__":
    main()

