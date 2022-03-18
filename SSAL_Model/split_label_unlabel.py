import pandas as pd

def split_label_unlabel(data,target,num_labeled,percentage_anomalies,seed):
    """Splits the data into a labeled set with num_labeled observations
    taking into consideration the percentage_anomalies in that set.
    The set needs to contain at least one anomaly.
    """
    num_anomaly = max(1, round(num_labeled*percentage_anomalies))
    num_normal = num_labeled-num_anomaly
    
    #Randomly select the labeled anomalies of the set
    anomalies = data[target.iloc[:,0] == 1]
    train_anomalies = anomalies.sample(n=num_anomaly, random_state=seed)
    #Randomly select the labeled normal observations of the set
    normal = data[target.iloc[:,0] == 0]
    train_normal=normal.sample(n=num_normal,random_state=seed)
    
    x_labeled = pd.concat([train_anomalies, train_normal]).sample(frac=1,
                                                                  random_state=seed)
    # Obtain the labels of the selected observations
    indexes = list(x_labeled.index)
    y_labeled = target.loc[indexes]
    
    # Attribute the remaining observations to the unlabeled set
    x_unlabeled = data.drop(indexes, axis=0)
    
    return x_labeled, y_labeled, x_unlabeled