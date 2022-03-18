import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Feats:

    def __init__(self, series_by_file, df_general, modes):
        self.df = pd.concat(series_by_file).reset_index(drop=True)
        self.general = df_general
        
        if 'diff' in modes and 'value' in modes:
            dict_feats = dict(table_feats('diff'), **table_feats('value'))
        elif 'diff' in modes:
            dict_feats = dict(table_feats('diff'))
        elif 'value' in modes:
            dict_feats = dict(table_feats('value'))
        
        self.table = pd.DataFrame.from_dict(dict_feats,orient='index',
                                            columns=['arguments','mode'])
        self.feats = pd.DataFrame(self.general['label'])

    def getAllFeats(self):
        """Extracts all features presented in the table of feats
        """
        for feature in self.table.index:
            self.extractFeature(feature)
            
    def extractFeature(self,name):
        """Extracts feature by its name as long as it belongs to the 
        table of feats.
        """
        if name in self.table.index:
            feat = self.table.loc[name]

            if feat['mode'] == 'division':
                self.feats[name] = division(self.general[feat['arguments'][0]],
                                            self.general[feat['arguments'][1]])
            elif feat['mode'] == 'division_sub':
                self.feats[name] = division_sub(self.general[feat['arguments'][0]],
                                                self.general[feat['arguments'][1]],
                                                self.general[feat['arguments'][2]])
            elif feat['mode'] == 'interval':
                self.feats[name] = interval(self.df['diff'],
                                            self.general[feat['arguments'][0]],
                                            self.general[feat['arguments'][1]],
                                            feat['arguments'][2])
            elif feat['mode'] == 'subtraction':
                self.feats[name] = subtraction(self.general[feat['arguments'][0]],
                                             self.general[feat['arguments'][1]])
            elif feat['mode'] == 'keep':
                self.feats[name] = self.general[feat['arguments'][0]]
        else:
            print('The feature with name', name, 'is not defined')

    def getFeats_dataframe(self):
        """Returns a dataframe with the extracted features
        """
        return self.feats


def drop_corr_feats(feats,classifier,seed):
    """Drops the highly correlated features according to the
    pre-specified classifier using 80% of the dataframe feats
    as training and 20% as validation
    """
    correlated_features = set()
    corr_matrix = feats.corr().abs()
    upper_tri = corr_matrix.where(np.triu(
        np.ones(corr_matrix.shape), k=1
        ).astype(np.bool))
    dict_of_f1=dict() #dictionary to store the f1_score of the features
    classif=classifier
    X_train, X_test, y_train, y_test = train_test_split(feats.drop(['label'],axis=1),
                                                        feats['label'],
                                                        test_size=0.2,
                                                        random_state=seed)
    for i in range(len(upper_tri.columns)):
        for j in range(i):
            if abs(upper_tri.iloc[j, i]) > 0.90:
                col1 = upper_tri.columns[i]
                col2 = upper_tri.columns[j]
                if col1 in dict_of_f1:
                    col1_f1 = dict_of_f1[col1]
                else:
                    classif.fit(X_train[[col1]], y_train)
                    y_pred = classif.predict(X_test[[col1]])
                    col1_f1 = metrics.f1_score(y_test, y_pred)
                    dict_of_f1[col1] = col1_f1
                if col2 in dict_of_f1:
                    col2_f1 = dict_of_f1[col2]
                else:
                    classif.fit(X_train[[col2]], y_train)
                    y_pred = classif.predict(X_test[[col2]])
                    col2_f1 = metrics.f1_score(y_test, y_pred)
                    dict_of_f1[col2] = col2_f1     

                colname = col1 if col1_f1 < col2_f1 else col2
                correlated_features.add(colname)
    print('Number of features dropped:', len(correlated_features))
    selected_features = feats.drop(labels=correlated_features, axis=1)
    return selected_features






def table_feats(mode):
    """Receives as input the mode that can be 'value' or 'diff'
    and returns a dictionary of features with the format 
    {name_feature:[list_arguments,function]}
    """
    dict_of_feats={
        # INTERVAL BASED FEATURES
        # Reach:sd-meanTot, Multiplier:3
        'meanTot_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanTot_'+mode,3],'interval'],
        'mean-medianTot_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean-medianTot_'+mode,3],'interval'],
        'medianTot_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianTot_'+mode,3],'interval'],
        'mean-maxTot_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean-maxTot_'+mode,3],'interval'],
        'meanAfter_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanAfter_'+mode,3],'interval'],
        'mean-medianBef_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean_medianBef_'+mode,3],'interval'],
        'medianAfter_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianAfter_'+mode,3],'interval'],
        'meanAfterwind_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanAfterwind_'+mode,3],'interval'],
        'mean-medianWind_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean_medianWind_'+mode,3],'interval'],
        'medianAfterwind_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianAfterwind_'+mode,3],'interval'],
        'meanLoc_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanLoc_'+mode,3],'interval'],
        'medianLoc_3sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianLoc_'+mode,3],'interval'],

        # Reach:sd-meanTot, Multiplier:4
        'meanTot_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanTot_'+mode,4],'interval'],
        'mean-medianTot_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean-medianTot_'+mode,4],'interval'],
        'medianTot_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianTot_'+mode,4],'interval'],
        'mean-maxTot_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean-maxTot_'+mode,4],'interval'],
        'meanAfter_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanAfter_'+mode,4],'interval'],
        'mean-medianBef_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean_medianBef_'+mode,4],'interval'],
        'medianAfter_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianAfter_'+mode,4],'interval'],
        'meanAfterwind_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanAfterwind_'+mode,4],'interval'],
        'mean-medianWind_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean_medianWind_'+mode,4],'interval'],
        'medianAfterwind_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianAfterwind_'+mode,4],'interval'],
        'meanLoc_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanLoc_'+mode,4],'interval'],
        'medianLoc_4sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianLoc_'+mode,4],'interval'],

        # Reach:sd-meanTot, Multiplier:5
        'meanTot_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanTot_'+mode,5],'interval'],
        'mean-medianTot_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean-medianTot_'+mode,5],'interval'],
        'medianTot_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianTot_'+mode,5],'interval'],
        'mean-maxTot_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean-maxTot_'+mode,5],'interval'],
        'meanAfter_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanAfter_'+mode,5],'interval'],
        'mean-medianBef_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean_medianBef_'+mode,5],'interval'],
        'medianAfter_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianAfter_'+mode,5],'interval'],
        'meanAfterwind_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanAfterwind_'+mode,5],'interval'],
        'mean-medianWind_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'mean_medianWind_'+mode,5],'interval'],
        'medianAfterwind_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianAfterwind_'+mode,5],'interval'],
        'meanLoc_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'meanLoc_'+mode,5],'interval'],
        'medianLoc_5sd-meanTot_'+mode:[['sd-meanTot_'+mode,'medianLoc_'+mode,5],'interval'],

        # Reach:sd-maxTot, Multiplier:1
        'meanTot_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanTot_'+mode,1],'interval'],
        'mean-medianTot_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean-medianTot_'+mode,1],'interval'],
        'medianTot_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianTot_'+mode,1],'interval'],
        'mean-maxTot_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean-maxTot_'+mode,1],'interval'],
        'meanAfter_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanAfter_'+mode,1],'interval'],
        'mean-medianBef_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean_medianBef_'+mode,1],'interval'],
        'medianAfter_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianAfter_'+mode,1],'interval'],
        'meanAfterwind_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanAfterwind_'+mode,1],'interval'],
        'mean-medianWind_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean_medianWind_'+mode,1],'interval'],
        'medianAfterwind_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianAfterwind_'+mode,1],'interval'],
        'meanLoc_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanLoc_'+mode,1],'interval'],
        'medianLoc_1sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianLoc_'+mode,1],'interval'],

        # Reach:sd-maxTot, Multiplier:2
        'meanTot_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanTot_'+mode,2],'interval'],
        'mean-medianTot_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean-medianTot_'+mode,2],'interval'],
        'medianTot_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianTot_'+mode,2],'interval'],
        'mean-maxTot_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean-maxTot_'+mode,2],'interval'],
        'meanAfter_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanAfter_'+mode,2],'interval'],
        'mean-medianBef_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean_medianBef_'+mode,2],'interval'],
        'medianAfter_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianAfter_'+mode,2],'interval'],
        'meanAfterwind_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanAfterwind_'+mode,2],'interval'],
        'mean-medianWind_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean_medianWind_'+mode,2],'interval'],
        'medianAfterwind_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianAfterwind_'+mode,2],'interval'],
        'meanLoc_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanLoc_'+mode,2],'interval'],
        'medianLoc_2sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianLoc_'+mode,2],'interval'],

        # Reach:sd-maxTot, Multiplier:3
        'meanTot_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanTot_'+mode,3],'interval'],
        'mean-medianTot_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean-medianTot_'+mode,3],'interval'],
        'medianTot_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianTot_'+mode,3],'interval'],
        'mean-maxTot_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean-maxTot_'+mode,3],'interval'],
        'meanAfter_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanAfter_'+mode,3],'interval'],
        'mean-medianBef_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean_medianBef_'+mode,3],'interval'],
        'medianAfter_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianAfter_'+mode,3],'interval'],
        'meanAfterwind_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanAfterwind_'+mode,3],'interval'],
        'mean-medianWind_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'mean_medianWind_'+mode,3],'interval'],
        'medianAfterwind_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianAfterwind_'+mode,3],'interval'],
        'meanLoc_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'meanLoc_'+mode,3],'interval'],
        'medianLoc_3sd-maxTot_'+mode:[['sd-maxTot_'+mode,'medianLoc_'+mode,3],'interval'],

        # Reach:sdBefore, Multiplier:3
        'meanTot_3sdBefore_'+mode:[['sdBefore_'+mode,'meanTot_'+mode,3],'interval'],
        'mean-medianTot_3sdBefore_'+mode:[['sdBefore_'+mode,'mean-medianTot_'+mode,3],'interval'],
        'medianTot_3sdBefore_'+mode:[['sdBefore_'+mode,'medianTot_'+mode,3],'interval'],
        'mean-maxTot_3sdBefore_'+mode:[['sdBefore_'+mode,'mean-maxTot_'+mode,3],'interval'],
        'meanAfter_3sdBefore_'+mode:[['sdBefore_'+mode,'meanAfter_'+mode,3],'interval'],
        'mean-medianBef_3sdBefore_'+mode:[['sdBefore_'+mode,'mean_medianBef_'+mode,3],'interval'],
        'medianAfter_3sdBefore_'+mode:[['sdBefore_'+mode,'medianAfter_'+mode,3],'interval'],
        'meanAfterwind_3sdBefore_'+mode:[['sdBefore_'+mode,'meanAfterwind_'+mode,3],'interval'],
        'mean-medianWind_3sdBefore_'+mode:[['sdBefore_'+mode,'mean_medianWind_'+mode,3],'interval'],
        'medianAfterwind_3sdBefore_'+mode:[['sdBefore_'+mode,'medianAfterwind_'+mode,3],'interval'],
        'meanLoc_3sdBefore_'+mode:[['sdBefore_'+mode,'meanLoc_'+mode,3],'interval'],
        'medianLoc_3sdBefore_'+mode:[['sdBefore_'+mode,'medianLoc_'+mode,3],'interval'],

        # Reach:sdBefore, Multiplier:4
        'meanTot_4sdBefore_'+mode:[['sdBefore_'+mode,'meanTot_'+mode,4],'interval'],
        'mean-medianTot_4sdBefore_'+mode:[['sdBefore_'+mode,'mean-medianTot_'+mode,4],'interval'],
        'medianTot_4sdBefore_'+mode:[['sdBefore_'+mode,'medianTot_'+mode,4],'interval'],
        'mean-maxTot_4sdBefore_'+mode:[['sdBefore_'+mode,'mean-maxTot_'+mode,4],'interval'],
        'meanAfter_4sdBefore_'+mode:[['sdBefore_'+mode,'meanAfter_'+mode,4],'interval'],
        'mean-medianBef_4sdBefore_'+mode:[['sdBefore_'+mode,'mean_medianBef_'+mode,4],'interval'],
        'medianAfter_4sdBefore_'+mode:[['sdBefore_'+mode,'medianAfter_'+mode,4],'interval'],
        'meanAfterwind_4sdBefore_'+mode:[['sdBefore_'+mode,'meanAfterwind_'+mode,4],'interval'],
        'mean-medianWind_4sdBefore_'+mode:[['sdBefore_'+mode,'mean_medianWind_'+mode,4],'interval'],
        'medianAfterwind_4sdBefore_'+mode:[['sdBefore_'+mode,'medianAfterwind_'+mode,4],'interval'],
        'meanLoc_4sdBefore_'+mode:[['sdBefore_'+mode,'meanLoc_'+mode,4],'interval'],
        'medianLoc_4sdBefore_'+mode:[['sdBefore_'+mode,'medianLoc_'+mode,4],'interval'],

        # Reach:sdBefore, Multiplier:5
        'meanTot_5sdBefore_'+mode:[['sdBefore_'+mode,'meanTot_'+mode,5],'interval'],
        'mean-medianTot_5sdBefore_'+mode:[['sdBefore_'+mode,'mean-medianTot_'+mode,5],'interval'],
        'medianTot_5sdBefore_'+mode:[['sdBefore_'+mode,'medianTot_'+mode,5],'interval'],
        'mean-maxTot_5sdBefore_'+mode:[['sdBefore_'+mode,'mean-maxTot_'+mode,5],'interval'],
        'meanAfter_5sdBefore_'+mode:[['sdBefore_'+mode,'meanAfter_'+mode,5],'interval'],
        'mean-medianBef_5sdBefore_'+mode:[['sdBefore_'+mode,'mean_medianBef_'+mode,5],'interval'],
        'medianAfter_5sdBefore_'+mode:[['sdBefore_'+mode,'medianAfter_'+mode,5],'interval'],
        'meanAfterwind_5sdBefore_'+mode:[['sdBefore_'+mode,'meanAfterwind_'+mode,5],'interval'],
        'mean-medianWind_5sdBefore_'+mode:[['sdBefore_'+mode,'mean_medianWind_'+mode,5],'interval'],
        'medianAfterwind_5sdBefore_'+mode:[['sdBefore_'+mode,'medianAfterwind_'+mode,5],'interval'],
        'meanLoc_5sdBefore_'+mode:[['sdBefore_'+mode,'meanLoc_'+mode,5],'interval'],
        'medianLoc_5sdBefore_'+mode:[['sdBefore_'+mode,'medianLoc_'+mode,5],'interval'],

        # Reach:sdWind, Multiplier:3
        'meanTot_3sdWind_'+mode:[['sdWind_'+mode,'meanTot_'+mode,3],'interval'],
        'mean-medianTot_3sdWind_'+mode:[['sdWind_'+mode,'mean-medianTot_'+mode,3],'interval'],
        'medianTot_3sdWind_'+mode:[['sdWind_'+mode,'medianTot_'+mode,3],'interval'],
        'mean-maxTot_3sdWind_'+mode:[['sdWind_'+mode,'mean-maxTot_'+mode,3],'interval'],
        'meanAfter_3sdWind_'+mode:[['sdWind_'+mode,'meanAfter_'+mode,3],'interval'],
        'mean-medianBef_3sdWind_'+mode:[['sdWind_'+mode,'mean_medianBef_'+mode,3],'interval'],
        'medianAfter_3sdWind_'+mode:[['sdWind_'+mode,'medianAfter_'+mode,3],'interval'],
        'meanAfterwind_3sdWind_'+mode:[['sdWind_'+mode,'meanAfterwind_'+mode,3],'interval'],
        'mean-medianWind_3sdWind_'+mode:[['sdWind_'+mode,'mean_medianWind_'+mode,3],'interval'],
        'medianAfterwind_3sdWind_'+mode:[['sdWind_'+mode,'medianAfterwind_'+mode,3],'interval'],
        'meanLoc_3sdWind_'+mode:[['sdWind_'+mode,'meanLoc_'+mode,3],'interval'],
        'medianLoc_3sdWind_'+mode:[['sdWind_'+mode,'medianLoc_'+mode,3],'interval'],

        # Reach:sdWind, Multiplier:4
        'meanTot_4sdWind_'+mode:[['sdWind_'+mode,'meanTot_'+mode,4],'interval'],
        'mean-medianTot_4sdWind_'+mode:[['sdWind_'+mode,'mean-medianTot_'+mode,4],'interval'],
        'medianTot_4sdWind_'+mode:[['sdWind_'+mode,'medianTot_'+mode,4],'interval'],
        'mean-maxTot_4sdWind_'+mode:[['sdWind_'+mode,'mean-maxTot_'+mode,4],'interval'],
        'meanAfter_4sdWind_'+mode:[['sdWind_'+mode,'meanAfter_'+mode,4],'interval'],
        'mean-medianBef_4sdWind_'+mode:[['sdWind_'+mode,'mean_medianBef_'+mode,4],'interval'],
        'medianAfter_4sdWind_'+mode:[['sdWind_'+mode,'medianAfter_'+mode,4],'interval'],
        'meanAfterwind_4sdWind_'+mode:[['sdWind_'+mode,'meanAfterwind_'+mode,4],'interval'],
        'mean-medianWind_4sdWind_'+mode:[['sdWind_'+mode,'mean_medianWind_'+mode,4],'interval'],
        'medianAfterwind_4sdWind_'+mode:[['sdWind_'+mode,'medianAfterwind_'+mode,4],'interval'],
        'meanLoc_4sdWind_'+mode:[['sdWind_'+mode,'meanLoc_'+mode,4],'interval'],
        'medianLoc_4sdWind_'+mode:[['sdWind_'+mode,'medianLoc_'+mode,4],'interval'],

        # Reach:sdWind, Multiplier:5
        'meanTot_5sdWind_'+mode:[['sdWind_'+mode,'meanTot_'+mode,5],'interval'],
        'mean-medianTot_5sdWind_'+mode:[['sdWind_'+mode,'mean-medianTot_'+mode,5],'interval'],
        'medianTot_5sdWind_'+mode:[['sdWind_'+mode,'medianTot_'+mode,5],'interval'],
        'mean-maxTot_5sdWind_'+mode:[['sdWind_'+mode,'mean-maxTot_'+mode,5],'interval'],
        'meanAfter_5sdWind_'+mode:[['sdWind_'+mode,'meanAfter_'+mode,5],'interval'],
        'mean-medianBef_5sdWind_'+mode:[['sdWind_'+mode,'mean_medianBef_'+mode,5],'interval'],
        'medianAfter_5sdWind_'+mode:[['sdWind_'+mode,'medianAfter_'+mode,5],'interval'],
        'meanAfterwind_5sdWind_'+mode:[['sdWind_'+mode,'meanAfterwind_'+mode,5],'interval'],
        'mean-medianWind_5sdWind_'+mode:[['sdWind_'+mode,'mean_medianWind_'+mode,5],'interval'],    
        'medianAfterwind_5sdWind_'+mode:[['sdWind_'+mode,'medianAfterwind_'+mode,5],'interval'],
        'meanLoc_5sdWind_'+mode:[['sdWind_'+mode,'meanLoc_'+mode,5],'interval'],
        'medianLoc_5sdWind_'+mode:[['sdWind_'+mode,'medianLoc_'+mode,5],'interval'],

        # Reach: sdLoc, Multiplier:1
        'meanTot_1sdLoc_'+mode:[['sdLoc_'+mode,'meanTot_'+mode,1],'interval'],
        'mean-medianTot_1sdLoc_'+mode:[['sdLoc_'+mode,'mean-medianTot_'+mode,1],'interval'],
        'medianTot_1sdLoc_'+mode:[['sdLoc_'+mode,'medianTot_'+mode,1],'interval'],
        'mean-maxTot_1sdLoc_'+mode:[['sdLoc_'+mode,'mean-maxTot_'+mode,1],'interval'],
        'meanAfter_1sdLoc_'+mode:[['sdLoc_'+mode,'meanAfter_'+mode,1],'interval'],
        'mean-medianBef_1sdLoc_'+mode:[['sdLoc_'+mode,'mean_medianBef_'+mode,1],'interval'],
        'medianAfter_1sdLoc_'+mode:[['sdLoc_'+mode,'medianAfter_'+mode,1],'interval'],
        'meanAfterwind_1sdLoc_'+mode:[['sdLoc_'+mode,'meanAfterwind_'+mode,1],'interval'],
        'mean-medianWind_1sdLoc_'+mode:[['sdLoc_'+mode,'mean_medianWind_'+mode,1],'interval'],
        'medianAfterwind_1sdLoc_'+mode:[['sdLoc_'+mode,'medianAfterwind_'+mode,1],'interval'],
        'meanLoc_1sdLoc_'+mode:[['sdLoc_'+mode,'meanLoc_'+mode,1],'interval'],
        'medianLoc_1sdLoc_'+mode:[['sdLoc_'+mode,'medianLoc_'+mode,1],'interval'],

        # NON-INTERVAL BASED FEATURES
        'meanLoc_vs_amplLoc_'+mode : [['meanLoc_'+mode,'amplLoc_'+mode,'maxLoc_'+mode],'division_sub'],
        'meanLoc_vs_maxLoc_'+mode:[['meanLoc_'+mode,'maxLoc_'+mode],'division'],
        'medianLoc_vs_amplLoc_'+mode : [['medianLoc_'+mode,'amplLoc_'+mode,'maxLoc_'+mode],'division_sub'], 
        'medianLoc_vs_maxLoc_'+mode:[['medianLoc_'+mode,'maxLoc_'+mode],'division'],
        'skewLoc_vs_skew-meanTot_'+mode: [['skewLoc_'+mode,'skew-meanTot_'+mode],'subtraction'],
        'skewLoc_'+mode: [['skewLoc_'+mode],'keep'],
        } 
    return dict_of_feats


def division(col1,col2):
    new_col = []
    for i in col1.index:
        den = col2[i]
        if den != 0:
            new_col.append(col1[i]/den)
        else:
            new_col.append(0.5)
    return new_col

def division_sub(col1,col2,col3):
    new_col = []
    for i in col1.index:
        num = col1[i]-(col3[i]-col2[i])
        den = col2[i]
        if den != 0:
            new_col.append(num/den)
        else:
            new_col.append(0.5)
    return new_col

def subtraction(col1,col2):
    new_col = []
    for i in col1.index:
        sub = col2[i]-col1[i]
        new_col.append(sub)
    return new_col

def interval(arr_set, reach_col, center_col, x):
    new_col = []
    for i in reach_col.index:
        result = 0
        center = center_col[i]
        reach = reach_col[i]
        arr = arr_set[i]
        for j in arr:
            if (j < center-x*reach or j > center+x*reach) and reach > 0:
                result += 1
        new_col.append(result/len(arr))
    return new_col
