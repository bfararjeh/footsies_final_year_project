from dataParsing import pullDataFromCSV
import pandas as pd
import numpy as np
import re, os, json


def normalise(df):

    # creates a config dictionary for dumping normalisation values later
    config = {}
    
    """
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
    """
    df["P1_left"]   = np.bitwise_and(
        np.right_shift(df["P1_currentInput"].astype(int), 2), 1)  # bit 2
    df["P1_right"]  = np.bitwise_and(
        np.right_shift(df["P1_currentInput"].astype(int), 1), 1)  # bit 1
    df["P1_attack"] = np.bitwise_and(
        df["P1_currentInput"].astype(int), 1)                     # bit 0
    df.drop(columns="P1_currentInput", inplace=True)    # drop the old column

    df["P2_left"]   = np.bitwise_and(
        np.right_shift(df["P2_currentInput"].astype(int), 2), 1)  # bit 2
    df["P2_right"]  = np.bitwise_and(
        np.right_shift(df["P2_currentInput"].astype(int), 1), 1)  # bit 1
    df["P2_attack"] = np.bitwise_and(
        df["P2_currentInput"].astype(int), 1)                     # bit 0
    df.drop(columns="P2_currentInput", inplace=True)    # drop the old column


    # simple normalisation of position value
    # since only the X position is relevant, the Y value is not included
    p1min_pos = df["P1_position"].min()
    p1max_pos = df["P1_position"].max()
    p2min_pos = df["P2_position"].min()
    p2max_pos = df["P2_position"].max()

    df["P1_position_norm"] = round((df["P1_position"] - p1min_pos) / (p1max_pos - p1min_pos), 3)
    df["P2_position_norm"] = round((df["P2_position"] - p2min_pos) / (p2max_pos - p2min_pos), 3)

    # same process for velocity
    p1min_vel = df["P1_velocity_x"].min()
    p1max_vel = df["P1_velocity_x"].max()
    p2min_vel = df["P2_velocity_x"].min()
    p2max_vel = df["P2_velocity_x"].max()

    df["P1_velocity_norm"] = round((df["P1_velocity_x"] - p1min_vel) / (p1max_vel - p1min_vel), 3)
    df["P2_velocity_norm"] = round((df["P2_velocity_x"] - p2min_vel) / (p2max_vel - p2min_vel), 3)

    df.drop(columns=["P1_position", 
                     "P2_position",
                     "P1_velocity_x",
                     "P2_velocity_x"], inplace=True)    # drop the old column

    config.update({
        "P2maxPosition":p2max_pos,
        "P2minPosition":p2min_pos,
        "P1maxPosition":p1max_pos,
        "P1minPosition":p1min_pos,
        "P2maxVelocity":p2max_vel,
        "P2minVelocity":p2min_vel,
        "P1maxVelocity":p1max_vel,
        "P1minVelocity":p1min_vel
    })


    # turns all boolean T/F values into 1/0 respectively
    df["P1_isDead"] = df["P1_isDead"].astype(int)
    df["P2_isDead"] = df["P2_isDead"].astype(int)
    df["P1_isAlwaysCancelable"] = df["P1_isAlwaysCancelable"].astype(int)
    df["P2_isAlwaysCancelable"] = df["P2_isAlwaysCancelable"].astype(int)
    df["P1_isInHitStun"] = df["P1_isInHitStun"].astype(int)
    df["P2_isInHitStun"] = df["P2_isInHitStun"].astype(int)


    # guardhealth is integer of range 0-3 inclusive. this normalises that
    # vital health is already integer range 0-1, so no normalising
    df["P1_guardpoint_norm"] = round(df["P1_guardHealth"] / 3, 3)
    df["P2_guardpoint_norm"] = round(df["P2_guardHealth"] / 3, 3)

    df.drop(columns=["P1_guardHealth",
                     "P2_guardHealth"], inplace=True)    # drop the old column


    # one hot encoding for currentActionID
    df = pd.get_dummies(df,
                        columns=["P1_currentActionID", "P2_currentActionID"],
                        prefix=["P1_moveID", "P2_moveID"])


    """
    these columns are dropped. this can be changed later, however these columns
    seem to only be relevant to drawing the fighter sprites
    """
    df.drop(columns=["P1_currentActionFrame",
                     "P1_currentActionFrameCount",
                     "P2_currentActionFrame",
                     "P2_currentActionFrameCount"], 
                     inplace=True)
    

    # these are dropped, as "isInHitStun" serves the same purpose
    df.drop(columns=["P1_currentHitStunFrame",
                     "P2_currentHitStunFrame"], 
                     inplace=True)



    # dict comprehension to convert values to floats instead of np.float64
    config = {n: float(config[n]) for n in config}

    """
    last piece of code uses a list comprehension to sort all columns such that
        P1 columns are on one side and P2 on the other
    """
    all_cols = df.columns.tolist()

    p1_cols = [col for col in all_cols if col.startswith("P1")]
    p2_cols = [col for col in all_cols if col.startswith("P2")]

    df = df[p1_cols + p2_cols]

    saveConfig(config)
    df.to_csv(os.path.join(os.path.dirname(__file__), 'normalisedOut.csv'))
    print("Data has been normalised")


def saveConfig(config):
    
    try:
        open("networkConfig.json", "r")

    except Exception:
        open("networkConfig.json", "a+")
    
    with open("networkConfig.json", "r+") as configFile:
        json.dump(config, configFile)
    


def main():
    normalise(pullDataFromCSV())


if __name__ == "__main__":
    main()

