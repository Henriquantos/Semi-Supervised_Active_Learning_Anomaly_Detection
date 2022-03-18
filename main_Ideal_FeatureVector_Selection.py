import glob
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from initialization import *
from preparation import *
from feats import *
from best_model import *


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
modes = ['diff'] # ['diff'], ['value'] or ['diff','value']
seed = 54391 

n_estimators = 20 
classifier = RandomForestClassifier(n_estimators,random_state=seed)
interval_of_features = (1,15) 
title = 'Diff-ov'+str(overlap)+'_'+str(classifier)[0]+str(n_estimators)

sfs = SFS(classifier,
          k_features=interval_of_features,
          forward=True,
          floating=False,
          verbose=0,
          scoring='f1',
          cv=10,
          n_jobs=-1)


# SEPARATE HOLDOUT AND TRAIN
train_data, holdout_data = separate_holdout(filenames,
                                            holdout_percentage,
                                            seed)


# PREPOCESSING
train_dividedSeries, train_generalFeats = divide_and_getGeneralFeats(train_data,
                                                                     window_size,
                                                                     overlap,
                                                                     seed,
                                                                     modes)


# EXTRACT FEATURES & ELIMINATE THE HIGHLY CORRELATED ONES
train_feats = Feats(train_dividedSeries, train_generalFeats, modes)
train_feats.getAllFeats()
train_feats_df = train_feats.getFeats_dataframe()

non_corr = drop_corr_feats(train_feats_df, classifier, seed)


# FIT THE SFS TO MAXIMIZE THE PERFORMANCE OF THE SETS OF FEATURES
sfs_fit = sfs.fit(non_corr.drop(['label'],axis=1),
                  non_corr['label'],
                  custom_feature_names=non_corr.drop(['label'], axis=1).columns)


# OBTAIN THE IDEAL FEATURE VECTOR AND THE CORRESPONDING F1-SCORE AND SAVE IT
selected_feats, f1_score = get_bestModel(sfs_fit,
                                         interval_of_features,
                                         title)
pd.DataFrame.from_dict(sfs_fit.get_metric_dict()).T.to_csv(title+'feats.csv')
