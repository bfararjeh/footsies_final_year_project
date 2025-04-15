import pandas as pd
import numpy as np
import re, os
'''
dummy line for testing

49: P1_INFO:currentInput(1)position(-1.16, 0.00)velocity_x(0)isDead(False)vitalHealth(1)guardHealth(3)currentActionID(105)currentActionFrame(10)currentActionFrameCount(21)isAlwaysCancelable(False)currentActionHitCount(1)currentHitStunFrame(0)isInHitStun(False)isAlwaysCancelable(False)P2_INFO:currentInput(2)position(0.93, 0.00)velocity_x(-0.5)isDead(False)vitalHealth(1)guardHealth(2)currentActionID(305)currentActionFrame(7)currentActionFrameCount(15)isAlwaysCancelable(False)currentActionHitCount(0)currentHitStunFrame(0)isInHitStun(False)isAlwaysCancelable(False)
'''

def parse_line(lineToParse):

    # splits the frame count as it only needs to be read once
    frame_match = re.match(r"(\d+): ", lineToParse)
    frame_number = int(frame_match.group(1))

    # splits the data into the p1 half and p2 half
    P1_data = lineToParse[lineToParse.index("P1_INFO:"):
                          lineToParse.index("P2_INFO:")]
    P2_data = lineToParse[lineToParse.index("P2_INFO:"):]

    '''defines a pattern with regex formatting to pull the values for each of
      the variables'''
    pattern = r"currentInput\((\d+)\)" + \
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

    P1matches = re.search(pattern, P1_data)
    P2matches = re.search(pattern, P2_data)

    '''goes through all the p1 and p2 matches and takes the value of each group
        as seen in the pattern above and adds it to a np array'''
    dataArray = np.array([])
    for i in range(2):
        if i == 1:
            currentPlayer = P1matches
        else:
            currentPlayer = P2matches

        for j in range(1,14):
            dataArray = np.append(dataArray, currentPlayer.group(j))
    
    # returns the frame number and data at that frame
    return frame_number, dataArray


def parse_dataset(dataset_path):

    os.chdir(dataset_path)
    dataRowList = []

    '''
    enumerates all the training files to grab an index and the file
    for each line in each file, parses the data and uses the return tuple
        to create a final array that contains the roundID, the framecounts, and
        the data
    then appends that array to a list
    after iterating through all files, the list is passed to the 
        create_dataframe function
    '''
    for round_ID, trainingFile in enumerate(os.listdir(dataset_path)):
        with open(trainingFile, "r") as openTrainingFile:
            for line in openTrainingFile:
                parsedData = parse_line(line)
                finalData = np.append(parsedData[0], parsedData[1])
                finalData = np.append(round_ID, finalData)
                dataRowList.append(finalData)
    
    create_dataframe(dataRowList)


def create_dataframe(data):
    
    # converts the data from a python list to a numpy list
    npData = np.asarray(data)

    '''
    creates a dataframe with 28 columns, one for each player and each
        variable and two for the roundID and frame number
    adds the data passed into the function to the dataframe
    '''
    extractedData = pd.DataFrame(
        data= npData,
        columns=[
            "round_ID",
            "frame_number",
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
            "P2_isInHitStun"])

    # extracts the data to a CSV for viewing
    extractedData.to_csv(os.path.join(os.path.dirname(__file__), 'out.csv'))


def main():

    '''
    bit messy, but this grabs the path of the training data with easy 
    changing of dataset path
    '''
    dataset_name = r"TrainingData\DATASET#2-NEW_AI"
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        dataset_name)
    
    # parses the dataset
    parse_dataset(dataset_path=dataset_path)
    print("Data succesfully extracted")


'''
function that can give me the ranges of values for normalisation needs
'''
def grabRanges():

    path = os.path.join(os.path.dirname(__file__), 'out.csv')

    extractedData = pd.read_csv(path)

    for label, content in extractedData.items():
        print(f"column name: {label}\n" + \
              "max value: {content.max()}\n" + \
              "min value: {content.min()}\n\n")

'''
function that pulls data from the "out.csv" file
'''
def pullDataFromCSV():

    path = os.path.join(os.path.dirname(__file__), 'out.csv')
    extractedData = pd.read_csv(path, index_col=0)

    return extractedData


if __name__ == "__main__":
    main()