import pandas as pd
import numpy as np
import re, os, json, time

'''
test line

"49: P1_INFO:currentInput(1)position(-1.16, 0.00)velocity_x(0)isDead(False)vitalHealth(1)guardHealth(3)currentActionID(105)currentActionFrame(10)currentActionFrameCount(21)isAlwaysCancelable(False)currentActionHitCount(1)currentHitStunFrame(0)isInHitStun(False)isAlwaysCancelable(False)P2_INFO:currentInput(2)position(0.93, 0.00)velocity_x(-0.5)isDead(False)vitalHealth(1)guardHealth(2)currentActionID(305)currentActionFrame(7)currentActionFrameCount(15)isAlwaysCancelable(False)currentActionHitCount(0)currentHitStunFrame(0)isInHitStun(False)isAlwaysCancelable(False)"
'''

class DataPreprocessor():
    '''
    This class handles all data preprocessing. It's methods:
    parseLine: parses a single string of data into a tuple
    parseDataset: iterates on a whole directory and parses all lines in all 
        files
    normalise: normalises a dataframe of any size
    normaliseDataset: normalises a dataset, generating new values for minmaxes
        and writes a config file
    pullDataset: pulls a normalised dataset from a csv
    parseAndNormalise: parses and then normalises a dataset
    '''
    def __init__(self):
        
        try:
            configPath = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "networkConfig.json")
            with open(configPath, "r") as rawConfig:
                self.config = json.load(rawConfig)

                self.p1min_pos, self.p1max_pos = self.config["minmax"]["P1_position"]
                self.p2min_pos, self.p2max_pos = self.config["minmax"]["P2_position"]
                self.p1min_vel, self.p1max_vel = self.config["minmax"]["P1_velocity"]
                self.p2min_vel, self.p2max_vel = self.config["minmax"]["P2_velocity"]

                self.normalisedColumnList = self.config["normalisedColumnList"]

        except Exception as e:
            print("Network config could not be read. Please re-normalise data.")

        # defines pattern list for parsing
        self.patternList = [
            r"currentInput\((\d+)\)",
            r"position\(([-\d.]+), 0.00\)",
            r"velocity_x\(([-\d.]+)\)",
            r"isDead\((True|False)\)",
            r"guardHealth\((\d+)\)",
            r"currentActionID\((\d+)\)",
            r"isAlwaysCancelable\((True|False)\)",
            r"isInHitStun\((True|False)\)",
            r"currentActionFrame\((\d+)\)",
            r"currentActionFrameCount\((\d+)\)",
        ]

        # defines expected column order via pattern list
        parsedColumns = ["round_ID", "frame_number"]
        for i in range(2):
            if i == 0:
                currentPlayer = "P1_"
            else:
                currentPlayer = "P2_"

            for pattern in self.patternList:
                columnName = pattern[:pattern.index("\\")]
                parsedColumns.append(currentPlayer + columnName)
        self.parsedColumns = parsedColumns

    def parseLine(self, rawString):
        '''
        This method parses a line and returns all data pulled from the line.
        The data parsed is dependant on the self.patternList of the class.
        Method takes one argument, "rawString", which is a line formatted as shown
            in the test line/as created by BattleCore.cs/UpdateFightState()
        Argument type: string
        Return type: integer, np.array, np.array
        '''

        # splits the frame count as it only needs to be read once
        frameNumber = int(re.match(r"(\d+): ", rawString).group(1))

        # splits the data into the p1 half and p2 half
        P1Data = rawString[rawString.index("P1_INFO:")
                            :rawString.index("P2_INFO:")]
        P2Data = rawString[rawString.index("P2_INFO:"):]

        P1Matches = np.array([])
        P2Matches = np.array([])

        try:
            for pattern in self.patternList:
                P1Matches = np.append(P1Matches, (re.search(pattern, P1Data)).group(1))
                P2Matches = np.append(P2Matches, (re.search(pattern, P2Data)).group(1))
        except Exception as e:
            print("Mismatch between client message and server parsing." + \
                  "\n Ensure that DataPreprocessor.patternList is updated.")

        return frameNumber, P1Matches, P2Matches

    def parseDataset(self, dataset_path):
        '''
        This method calls the "parseLine" method on each line in each file in
            a given directory.
        It enumerates all training files to grab an index (the round ID) and 
            file. For each line, it combines the round ID, frame number, and
            any parsed data into one array entry.
        Argument type: path
        Return type: dataframe
        '''
        start = time.perf_counter()

        os.chdir(dataset_path)
        dataRowList = []

        for roundID, trainingFile in enumerate(os.listdir(dataset_path)):
            with open(trainingFile, "r") as openTrainingFile:
                for line in openTrainingFile:
                    frameNumber, P1Matches, P2Matches = self.parseLine(line)
                    finalData = np.append(roundID, frameNumber)
                    finalData = np.append(finalData, [P1Matches, P2Matches])
                    dataRowList.append(finalData)

        # creates dataframe object using self.parsed columns and datarowlist
        extractedData = pd.DataFrame(
            data=np.asarray(dataRowList),
            columns=self.parsedColumns)
        
        print(f"Parsing: {(time.perf_counter() - start):.2f}s")
        return extractedData

    def normalise(self, rawDataFrame):
        '''
        This method runs normalisation on a dataframe. Can be used on a 
            dataframe of any size, allowing for use on single lines and entire
            datasets.
        List of all normalisations that take place:
            currentInput - bitwise decomposition into three columns (for each button)
            position/velocity_x - minmax normalisation
            isDead, isAlwaycCancelable, isInHitStun - converts T/F to 1/0
            guardHealth - normalises into range [0,1]
            currentActionID - one hot encodes all moves
        Unused/columns pre normalisation are removed then columns are sorted.
        Argument type: dataframe
        Return type: dataframe
        '''
        df = rawDataFrame

        # decomposing bitwise mask of currentInput to 3 seperate columns
        df["P1_attack"]   = np.bitwise_and(
            np.right_shift(df["P1_currentInput"].astype(int), 2), 1)  # bit 2
        df["P1_right"]  = np.bitwise_and(
            np.right_shift(df["P1_currentInput"].astype(int), 1), 1)  # bit 1
        df["P1_left"] = np.bitwise_and(
            df["P1_currentInput"].astype(int), 1)                     # bit 0
    
        # normalisation of position and velocity
        df["P1_position_norm"] = round((df["P1_position"].astype(float) - self.p1min_pos) / (self.p1max_pos - self.p1min_pos), 3)
        df["P2_position_norm"] = round((df["P2_position"].astype(float) - self.p2min_pos) / (self.p2max_pos - self.p2min_pos), 3)

        df["distance"] = df["P2_position"].astype(float)  - df["P1_position"].astype(float) 
        df["distance_norm"] = df["P2_position_norm"] - df["P1_position_norm"]
        df["distance_delta"] = df["distance_norm"].diff()

        def label_direction(delta):
            if pd.isna(delta):
                return 0  # first frame
            elif delta > 0.01:
                return -1   # retreating
            elif delta < -0.01:
                return 1  # approaching
            else:
                return 0   # neutral / idle
        df["approach_retreat"] = df["distance_delta"].apply(label_direction)

        df["in_threat_range"] = pd.cut(
            df["distance"],
            bins=[0, 2.34, 2.54, 8.6],
            labels=[1.0, 0.5, 0.0],
            right=False
        ).astype(float)

        df["P1_is_cornered"] = pd.cut(
            df["P1_position"].astype(float) ,
            bins=[-float('inf'), -3.3, float("inf")],
            labels=[1, 0],
            right=int
        ).astype(float)

        df["P2_is_cornered"] = pd.cut(
            df["P2_position"].astype(float) ,
            bins=[-float('inf'), 3.3, float("inf")],
            labels=[0, 1],
            right=False
        ).astype(int)

        df["P1_velocity_norm"] = round((df["P1_velocity_x"].astype(float) - self.p1min_vel) / (self.p1max_vel - self.p1min_vel), 3)
        df["P2_velocity_norm"] = round((df["P2_velocity_x"].astype(float) - self.p2min_vel) / (self.p2max_vel - self.p2min_vel), 3)

        # turns all boolean T/F values into 1/0 respectively
        for col in ["P1_isDead", 
                    "P1_isAlwaysCancelable", 
                    "P1_isInHitStun", 
                    "P2_isDead", 
                    "P2_isAlwaysCancelable", 
                    "P2_isInHitStun"]:
            df[col] = df[col].map({'False':0, 'True':1})

        # guardhealth is integer of range 0-3 inclusive. this normalises that
        df["P1_guardpoint_norm"] = round(df["P1_guardHealth"].astype(float) / 3, 3)
        df["P2_guardpoint_norm"] = round(df["P2_guardHealth"].astype(float) / 3, 3)

        # changing how move id is stored/checked

        # MOVEMENT
        # stand 	    =	0
        # forward       =	1
        # backward	    =	2
        # forward dash	=	10
        # back dash		=	11

        # ATTACKS
        # n attack  	=	100
        # b attack	    =	105
        # n special		=	110
        # b special     =	115

        # GETTING HIT
        # damage 	    = 	200
        # prox guard    =	350
        # guard stand   = 	301
        # guard crouch  =   305
        # guard break   =   310

        # dead  	    =	500
        # win           =   510

        # the following columns are derived features 
        # trying to figure out how to improve the network
        df["P1_currentActionID"] = df["P1_currentActionID"].astype(int)
        df["P2_currentActionID"] = df["P2_currentActionID"].astype(int)

        df["P1_n_attack_punish"] = (
            (df["distance"] >= 2.34) & 
            (df["distance"] <= 3.18) & 
            (df["P2_currentActionID"] == 105)
        ).astype(int) | (
            (df["distance"] >= 2.54) & 
            (df["distance"] <= 3.38) & 
            (df["P2_currentActionID"] == 100)
        ).astype(int)

        df["P1_b_attack_punish"] = (
            (df["distance"] >= 2.34) & 
            (df["distance"] <= 3.0) & 
            (df["P2_currentActionID"] == 105)
        ).astype(int) | (
            (df["distance"] >= 2.54) & 
            (df["distance"] <= 3.20) & 
            (df["P2_currentActionID"] == 100)
        ).astype(int)
        
        df["P1_normal_landed"] = (
            ((df["P1_currentActionID"] == 100) | (df["P1_currentActionID"] == 105)) &
            (df["P2_currentActionID"] == 200)
        ).astype(int)

        df["P1_is_guarding"] = (
            (df["P1_currentActionID"] == 350) |
            (df["P1_currentActionID"] == 301) |
            (df["P1_currentActionID"] == 305)
        ).astype(int)

        df["P1_is_hit"] = (
            df["P1_currentActionID"] == 200
        ).astype(int)


        df["P2_n_attack_punish"] = (
            (df["distance"] >= 2.34) & 
            (df["distance"] <= 3.18) & 
            (df["P1_currentActionID"] == 105)
        ).astype(int) | (
            (df["distance"] >= 2.54) & 
            (df["distance"] <= 3.38) & 
            (df["P1_currentActionID"] == 100)
        ).astype(int)

        df["P2_b_attack_punish"] = (
            (df["distance"] >= 2.34) & 
            (df["distance"] <= 3.0) & 
            (df["P1_currentActionID"] == 105)
        ).astype(int) | (
            (df["distance"] >= 2.54) & 
            (df["distance"] <= 3.20) & 
            (df["P1_currentActionID"] == 100)
        ).astype(int)
        
        df["P2_normal_landed"] = (
            ((df["P2_currentActionID"] == 100) | (df["P2_currentActionID"] == 105)) &
            (df["P1_currentActionID"] == 200)
        ).astype(int)

        df["P2_is_guarding"] = (
            (df["P2_currentActionID"] == 350) |
            (df["P2_currentActionID"] == 301) |
            (df["P2_currentActionID"] == 305)
        ).astype(int)

        df["P2_is_hit"] = (
            df["P2_currentActionID"] == 200
        ).astype(int)


        df["P1_frame_advantage"] = df["P1_currentActionFrameCount"].astype(int) - df["P1_currentActionFrame"].astype(int)
        df["P2_frame_advantage"] = df["P2_currentActionFrameCount"].astype(int) - df["P1_currentActionFrame"].astype(int)

        df['P1_frame_advantage'] = df['P1_frame_advantage'].mask(df["P1_isAlwaysCancelable"].astype(int) == 1, 0)
        df['P2_frame_advantage'] = df['P2_frame_advantage'].mask(df["P2_isAlwaysCancelable"].astype(int) == 1, 0)

        df["P1_frame_advantage"] = df["P2_frame_advantage"].astype(int) - df["P1_frame_advantage"].astype(int)
        df["P2_frame_advantage"] = df["P1_frame_advantage"].astype(int) - df["P2_frame_advantage"].astype(int)

        df.loc[df['P1_frame_advantage'] < 0, 'P1_frame_advantage'] = -1
        df.loc[df['P1_frame_advantage'] == 0, 'P1_frame_advantage'] = 0
        df.loc[df['P1_frame_advantage'] > 0, 'P1_frame_advantage'] = 1

        df.loc[df['P2_frame_advantage'] < 0, 'P2_frame_advantage'] = -1
        df.loc[df['P2_frame_advantage'] == 0, 'P2_frame_advantage'] = 0
        df.loc[df['P2_frame_advantage'] > 0, 'P2_frame_advantage'] = 1


        # drops all replaced/unecessary columns
        df.drop(columns=["distance",
                         "distance_delta",
                         "P1_currentInput",
                         "P1_position",
                         "P1_velocity_x",
                         "P1_guardHealth",
                         "P1_currentActionID",
                         "P1_currentActionFrame",
                         "P1_currentActionFrameCount",
                         "P2_currentInput",
                         "P2_position",
                         "P2_velocity_x",
                         "P2_guardHealth",
                         "P2_currentActionID"
                         "P2_currentActionFrame",
                         "P2_currentActionFrameCount",], inplace=True, errors="ignore")
        
        # moves all p2 columns to right side
        allColumns = df.columns.tolist()

        p2Cols = [col for col in allColumns if col.startswith("P2")]
        otherCols = [col for col in allColumns if col not in p2Cols] 
        df = df[otherCols + p2Cols]
    
        return df
    
    def normaliseDataset(self, rawDataset, configPath):
        '''
        This method runs normalisation on an entire dataset. This method exists
            as the normalise() method uses information from the config file, 
            this function creates a new config file for "fresh" normalising.
        The normalised dataframe is then saved to a csv file.
        Argument type: dataframe
        '''

        # pulls minmax values for relevant columns
        self.p1min_pos = rawDataset["P1_position"].astype(float).min()
        self.p1max_pos = rawDataset["P1_position"].astype(float).max()
        self.p2min_pos = rawDataset["P2_position"].astype(float).min()
        self.p2max_pos = rawDataset["P2_position"].astype(float).max()
    
        self.p1min_vel = rawDataset["P1_velocity_x"].astype(float).min()
        self.p1max_vel = rawDataset["P1_velocity_x"].astype(float).max()
        self.p2min_vel = rawDataset["P2_velocity_x"].astype(float).min()
        self.p2max_vel = rawDataset["P2_velocity_x"].astype(float).max()

        # creates empty dict for normalisation val dumping
        config = {
            "minmax": {
                "P1_position": [self.p1min_pos, self.p1max_pos],
                "P2_position": [self.p2min_pos, self.p2max_pos],
                "P1_velocity": [self.p1min_vel, self.p1max_vel],
                "P2_velocity": [self.p2min_vel, self.p2max_vel],
            }
        }

        start = time.perf_counter()
        # calling the normalise function
        normalisedDF = self.normalise(rawDataset)
        print(f"Normalisation: {(time.perf_counter() - start):.2f}s")

        # creating a config entry for all columns post normalisation
        config["normalisedColumnList"] = normalisedDF.columns.tolist()

        dfSavePath = os.path.join(os.path.dirname(__file__), 'normalisedDatasetExperimental.csv')
        normalisedDF.to_csv(dfSavePath)

        with open(os.path.join(configPath, "networkConfigExperimental.json"), "w") as configFile:
            json.dump(config, configFile)
        
        print(f"Data normalised to {dfSavePath} and config saved.")
    
    def pullDataset(self):
        '''
        This method pulls the normalised dataframe from the csv and returns it.
        Will fail if no file exists.
        Return type: dataframe
        '''

        path = os.path.join(os.path.dirname(__file__), 'normalisedDataset.csv')
        return pd.read_csv(path, index_col=0)

    def parseAndNormalise(self, datasetPath, configPath):
        '''
        Parses and then normalises the dataset.
        Argument type: path
        '''

        self.normaliseDataset(self.parseDataset(datasetPath), configPath)

    def normaliseLiveInput(self, rawString):
        '''
        Normalises a single line and returns the value.
        Argument type: string
        Return type: dataframe
        '''
        
        frameNumber, P1Matches, P2Matches = self.parseLine(rawString)
        parsedLine = np.append(0, frameNumber)
        parsedLine = np.append(parsedLine, [P1Matches, P2Matches])

        df = pd.DataFrame(
            data=np.asarray([parsedLine]),
            columns=self.parsedColumns)
        
        return self.normalise(df)
    

def main():
    '''
    Function that will preprocess all data.
    To be used only by the developer.
    Calls when class is run as file and preprocesses all data.
    '''

    myPreprocessor = DataPreprocessor()

    datasetName = r"TrainingData\DATASET#3-COMBINED"
    datasetPath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        datasetName)
    configPath = os.getcwd()

    myPreprocessor.parseAndNormalise(datasetPath, configPath)

if __name__ == "__main__":
    main()