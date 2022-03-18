#IMPORT LIBRARIES
import glob
from sklearn.ensemble import RandomForestClassifier

from initialization import *
from preprocess import *
from split_label_unlabel import *
from ssal import *


# IMPORT DATA - available for download at
# https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70
path1 #= path of folder 'A1Benchmark'
path2 #= path of folder 'A2Benchmark'
path3 #= path of folder 'A3Benchmark'
path4 #= path of folder 'A4Benchmark'

filenames = [glob.glob(path1 + "/*.csv"),
             glob.glob(path2 + "/*.csv"),
             glob.glob(path3 + "/*.csv"),
             glob.glob(path4 + "/*.csv")]


# DEFINITION OF PARAMETERS
holdout_percentage = 0.1 
window_size = 24 
overlap = True
seed = 54391
classifier = RandomForestClassifier(20, random_state=seed)
modes = ['diff'] # ['diff'], ['value'] or ['diff','value']
st = 'rl' # 'rl' or 'nrl'
qs = 'R' # 'R','Unc','Ut','Unc&Ut1','Unc&Ut2','Unc&Ut1_variation' or 'Unc&Ut2_variation'

num_labeled = 2 # Number of inialially labelled training observations
percentage_anomalies = 0.5


# SEPARATE HOLDOUT AND TRAIN
train_data, holdout_data = separate_holdout(filenames,
                                            holdout_percentage,
                                            seed)


# PREPROCESS
X_train, y_train = preprocess(train_data, window_size, overlap, seed, modes)
X_test, y_test = preprocess(holdout_data, window_size, overlap, seed, modes)



# DIVIDE TRAINING SET INTO LABELED AND UNLABELED SUBSETS
X_labeled, y_labeled,\
    X_unlabeled = split_label_unlabel(X_train,
                                      y_train,
                                      num_labeled=num_labeled,
                                      percentage_anomalies=percentage_anomalies,
                                      seed=seed)


# APPLY SEMI SUPERVISED ACTIVE LEARNING MODEL
result = ssal(X_labeled,
              y_labeled,
              X_unlabeled,
              y_train,
              X_test,
              y_test, 
              limit=1,
              classifier=classifier,
              seed=seed,
              qs=qs,
              st=st)


# EXPORT RESULTS
title = str(num_labeled)+'_'+qs+'_'+st
result.to_excel(title+"xlsx")
