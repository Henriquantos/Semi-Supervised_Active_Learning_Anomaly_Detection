from feats import *
from preparation_part2 import *

# Features selected by the Random Forest Classifier with 20 decision trees
selected_feats=['mean-maxTot_3sd-meanTot_diff',
                'mean-medianWind_3sd-meanTot_diff', 
                'mean-maxTot_4sd-meanTot_diff', 
                'meanAfterwind_5sd-meanTot_diff',
                'mean-medianBef_1sd-maxTot_diff',
                'medianLoc_3sdBefore_diff',
                'meanLoc_5sdWind_diff',
                'meanTot_1sdLoc_diff', 
                'mean-maxTot_1sdLoc_diff',
                'meanAfterwind_1sdLoc_diff', 
                'skewLoc_vs_skew-meanTot_diff']


def preprocess(set_series, window_size, overlap, seed, modes):
    """Applies the preprocess to the set of Time Series and returns the
    extracted Ideal Feature Vectors for all stretches along with the
    respective labels
    """
    dividedSeries, generalFeats = divide_and_getGeneralFeats(set_series,
                                                             window_size,
                                                             overlap,
                                                             seed,
                                                             modes) 
    feats = Feats(dividedSeries, generalFeats, modes)
    for feat in selected_feats:
        feats.extractFeature(feat)
        
    feats_df = feats.getFeats_dataframe()
    
    X = feats_df.drop(['label'], axis=1)
    y = feats_df[['label']]

    return X,y
