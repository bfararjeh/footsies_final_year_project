from dataParsing import pullDataFromCSV
import pandas as pd
import numpy as np
import re, os, json

class Normaliser():
    def __init__(self):
        self.normalisedDataset = ""
        
        try:
            with open("networkConfig.json", "r") as rawConfig:
                self.config = json.load(rawConfig)

                self.p1min_pos, self.p1max_pos = self.config["minmax"]["P1_position"]
                self.p2min_pos, self.p2max_pos = self.config["minmax"]["P2_position"]
                self.p1min_vel, self.p1max_vel = self.config["minmax"]["P1_velocity"]
                self.p2min_vel, self.p2max_vel = self.config["minmax"]["P2_velocity"]

                self.onehot_columns = self.config["onehot_columns"]

        except Exception as e:
            print("Network config could not be read. Please re-normalise data.")
            print(e)

        self.defaultParsedColumns = [
            "P1_currentInput",
            "P1_position",
            "P1_velocity_x",
            "P1_isDead",
            "P1_vitalHealth",
            "P1_guardHealth",
            "P1_currentActionID",
            "P1_currentActionFrame",
            "P1_currentActionFrameCount",
            "P1_isAlwaysCancelable",
            "P1_currentActionHitCount",
            "P1_currentHitStunFrame",
            "P1_isInHitStun",
            "P2_currentInput",
            "P2_position",
            "P2_velocity_x",
            "P2_isDead",
            "P2_vitalHealth",
            "P2_guardHealth",
            "P2_currentActionID",
            "P2_currentActionFrame",
            "P2_currentActionFrameCount",
            "P2_isAlwaysCancelable",
            "P2_currentActionHitCount",
            "P2_currentHitStunFrame",
            "P2_isInHitStun"]


    def normaliseDataset(self, df):

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
        df["P1_attack"]   = np.bitwise_and(
            np.right_shift(df["P1_currentInput"].astype(int), 2), 1)  # bit 2
        df["P1_right"]  = np.bitwise_and(
            np.right_shift(df["P1_currentInput"].astype(int), 1), 1)  # bit 1
        df["P1_left"] = np.bitwise_and(
            df["P1_currentInput"].astype(int), 1)                     # bit 0
        df.drop(columns="P1_currentInput", inplace=True)    # drop the old column

        # we drop the p2 input column here as the model should not be able to read
        #   inputs
        df.drop(columns="P2_currentInput", inplace=True)


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


        # turns all boolean T/F values into 1/0 respectively
        for col in ["P1_isDead", 
                    "P2_isDead", 
                    "P1_isAlwaysCancelable", 
                    "P2_isAlwaysCancelable", 
                    "P1_isInHitStun", 
                    "P2_isInHitStun"]:
            df[col] = df[col].astype(int)


        # guardhealth is integer of range 0-3 inclusive. this normalises that
        # vital health is already integer range 0-1, so no normalising
        df["P1_guardpoint_norm"] = round(df["P1_guardHealth"] / 3, 3)
        df["P2_guardpoint_norm"] = round(df["P2_guardHealth"] / 3, 3)

        df.drop(columns=["P1_guardHealth",
                        "P2_guardHealth"], inplace=True)    # drop the old column


        # one hot encoding for currentActionID
        df = pd.get_dummies(df,
                            columns=["P1_currentActionID", "P2_currentActionID"],
                            prefix=["P1_moveID", "P2_moveID"],
                            dtype=int)
        df["P2_moveID_11"] = 0 # because the ai didnt backdash once lmao


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

        # these are droppes, as vitalHealth serves the same purpose as isDead
        df.drop(columns=["P1_vitalHealth",
                        "P2_vitalHealth"], 
                        inplace=True)


        """
        last piece of code uses a list comprehension to sort all columns such that
            P1 columns are on one side and P2 on the other
        """
        all_cols = df.columns.tolist()
        p2_cols = [col for col in all_cols if col.startswith("P2")]
        other_cols = [col for col in all_cols if col not in p2_cols]

        df = df[other_cols + p2_cols]

        # saving the config file
        config = {
            "minmax": {
                "P1_position": [float(p1min_pos), float(p1max_pos)],
                "P2_position": [float(p2min_pos), float(p2max_pos)],
                "P1_velocity": [float(p1min_vel), float(p1max_vel)],
                "P2_velocity": [float(p2min_vel), float(p2max_vel)]},

            "onehot_columns": df.columns.tolist()
            }


        df.to_csv(os.path.join(os.path.dirname(__file__), 'normalisedOut.csv'))
        print("Data has been normalised.")

        with open("networkConfig.json", "w+") as configFile:
            json.dump(config, configFile)
            print("Normalisation config saved.")


    def normaliseLine(self, line):
        """
        normalises a single row dataframe with the config
        ensures same format as training data
        returns row as df with all features in same column order.
        """

        # turns the np.ndarray, which is the returned type of parse_line() into
        #   a df
        dfLine = pd.DataFrame([line], columns=self.defaultParsedColumns)

        # bitmask doesnt need to be decomposed as those are targets for 
        #   prediction
        dfLine.drop(columns=["P1_currentInput", 
                             "P2_currentInput"], inplace=True)

        # normalising position
        # now we use self variables pulled from the config
        dfLine["P1_position_norm"] = round(
            (dfLine["P1_position"].astype(float) - self.p1min_pos) / (self.p1max_pos - self.p1min_pos), 3)
        dfLine["P2_position_norm"] = round(
            (dfLine["P2_position"].astype(float) - self.p2min_pos) / (self.p2max_pos - self.p2min_pos), 3)

        # same thing w velocity
        dfLine["P1_velocity_norm"] = round(
            (dfLine["P1_velocity_x"].astype(float) - self.p1min_vel) / (self.p1max_vel - self.p1min_vel), 3)
        dfLine["P2_velocity_norm"] = round(
            (dfLine["P2_velocity_x"].astype(float) - self.p2min_vel) / (self.p2max_vel - self.p2min_vel), 3)

        dfLine.drop(columns=["P1_position", 
                             "P2_position", 
                             "P1_velocity_x", 
                             "P2_velocity_x"], inplace=True)

        # convert booleans
        for col in ["P1_isDead", 
                    "P2_isDead", 
                    "P1_isAlwaysCancelable", 
                    "P2_isAlwaysCancelable",
                    "P1_isInHitStun", 
                    "P2_isInHitStun"]:
            dfLine[col] = dfLine[col].astype(bool).astype(int)

        # THE CULPRIT
        dfLine["P1_currentActionHitCount"] = dfLine["P1_currentActionHitCount"].astype("Int64")
        dfLine["P2_currentActionHitCount"] = dfLine["P2_currentActionHitCount"].astype("Int64")

        # normalise guard health
        dfLine["P1_guardpoint_norm"] = round(dfLine["P1_guardHealth"].astype(float) / 3, 3)
        dfLine["P2_guardpoint_norm"] = round(dfLine["P2_guardHealth"].astype(float) / 3, 3)
        dfLine.drop(columns=["P1_guardHealth", "P2_guardHealth"], inplace=True)

        # one hot encoding
        dfLine = pd.get_dummies(dfLine,
                                columns=["P1_currentActionID", "P2_currentActionID"],
                                prefix=["P1_moveID", "P2_moveID"],
                                dtype=int)

        # dropping irrelevant columns
        # you should move this to data parsing
        drop_cols = ["P1_currentActionFrame", 
                     "P1_currentActionFrameCount",
                    "P2_currentActionFrame", 
                    "P2_currentActionFrameCount",
                    "P1_currentHitStunFrame", 
                    "P2_currentHitStunFrame",
                    "P1_vitalHealth", 
                    "P2_vitalHealth"]
        dfLine.drop(columns=[col for col in drop_cols if col in dfLine.columns], inplace=True)

        # this uses the config file to ensure the single line df has the same
        #   amount of columns as the training data set
        for col in self.onehot_columns:
            if col not in dfLine.columns:
                dfLine[col] = 0

        # this reorders data to the same order as the training set
        dfLine = dfLine[self.onehot_columns]

        dfLine.drop(columns=["round_ID",
                             "frame_number",
                             "P1_attack",
                             "P1_left",
                             "P1_right"], inplace=True, errors="ignore")
        
        return dfLine


    def pullNormalisedDataFromCSV(self):

        path = os.path.join(os.path.dirname(__file__), 'normalisedOut.csv')
        extractedData = pd.read_csv(path, index_col=0)

        return extractedData
    
    def runNormalisation(self):
        
        self.normaliseDataset(pullDataFromCSV())


def main():

    myNormaliser = Normaliser()
    myNormaliser.runNormalisation()


if __name__ == "__main__":
    main()