from dataParsing import pullDataFromCSV
import pandas as pd
import numpy as np
import re, os


def oneHotEncode(dataframe):

    print(dataframe)
    
    oneHotDataFrame = pd.get_dummies(dataframe, 
                                     columns=["P1_currentActionID", "P2_currentActionID"],
                                     prefix=["P1_moveID", "P2_moveID"])
    
    print(oneHotDataFrame)







def main():
    oneHotEncode(pullDataFromCSV())


if __name__ == "__main__":
    main()

