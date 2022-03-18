import pandas as pd
import random

def separate_holdout(filenames, holdout_percentage, seed):
    """Randomly selects holdout_percentage of the files belonging to
    filenames (that is a list of lists with each list representing a folder),
    being this selection made by folder. Imports the data from the
    specified files and returns the data used for train and for test.
    """
    random.seed(seed)

    holdout_files12 = random.sample(filenames[0],
                                    round(len(filenames[0])*holdout_percentage))
    holdout_files12.extend(
        random.sample(filenames[1], round(len(filenames[1])*holdout_percentage))
        )

    train_files12 = [
        element for element in (filenames[0]+filenames[1]) 
        if element not in holdout_files12
        ]
      
    holdout_files34 = random.sample(filenames[2],
                                    round(len(filenames[2])*holdout_percentage))
    holdout_files34.extend(
        random.sample(filenames[3], round(len(filenames[3])*holdout_percentage))
        )
    
    train_files34 = [
        element for element in (filenames[2]+filenames[3]) 
        if element not in holdout_files34
        ]

    train_data = import_data(train_files12, train_files34)
    holdout_data = import_data(holdout_files12, holdout_files34)

    return train_data, holdout_data


def import_data(file_list12, file_list34):
    """Imports the data from the files of the specified folders
    """
    data=[]
    for filename in file_list12:
        data.append(pd.read_csv(filename).loc[:, ['value', 'is_anomaly']])
    for filename in file_list34:
        data.append(pd.read_csv(filename).loc[:, ['value', 'anomaly']].rename(columns={'anomaly': 'is_anomaly'}))
    return data