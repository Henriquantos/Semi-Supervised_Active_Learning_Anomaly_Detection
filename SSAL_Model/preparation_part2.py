import numpy as np
import pandas as pd
from scipy.stats import skew


def divide_and_getGeneralFeats(set_series, window_size, overlap, seed, modes):
    """Eliminates the initialization outliers, obtains the first differences,
    divides the Time Series and obtains feature regarding the original Time 
    Series and the stretch itself. Returns the stretches resulting from the
    division of the Time Series and the features of them.
    """
    eliminate_initial_outliers(set_series, window_size)
    create_diffs(set_series)
    print('The differences were obtained')
    divided_series = divideTS(set_series, window_size, overlap)
    print('The Time Series were divided')
    
    if 'diff' in modes and 'value' in modes:
        generalFeats_diff = all_generalFeats(divided_series, window_size,
                                             overlap, seed, 'diff')
        generalFeats_value = all_generalFeats(divided_series, window_size,
                                              overlap, seed, 'value')
        generalFeats_value.drop(['label'], inplace=True, axis=1)
        generalFeats = pd.concat([generalFeats_diff, generalFeats_value],
                                 axis=1)
        print('General Feats obtained')
        return divided_series, generalFeats
    
    elif 'diff' in modes:
        generalFeats_diff = all_generalFeats(divided_series, window_size,
                                             overlap, seed, 'diff')
        print('General Feats obtained')
        return divided_series, generalFeats_diff
    
    elif 'value' in modes:
        generalFeats_value = all_generalFeats(divided_series, window_size,
                                              overlap, seed, 'value')
        print('General Feats obtained')
        return divided_series, generalFeats_value
    
    else:
        print('Modes passed are invalid')

        
# Auxiliar function of divide_and_getGeneralFeats
def eliminate_initial_outliers(set_series, window_size):
    """Receives a set of Time Series and the window size used in the 
    slicing and drops, inplace, the initial outliers that are caused 
    by the initialization of the sensors.
    """
    n_dropped = 0
    
    for series in set_series: 
        block = series[3:window_size]['value']
        mean = np.mean(block)
        sd = np.std(block)
        
        if (series['value'].iloc[0] == 0.0 and min(series[3:100]['value']) > 0):
            series.drop(index=series.index[0], axis=0, inplace=True)
            n_dropped += 1
        
        while (series['value'].iloc[0] > mean+3*sd) or (series['value'].iloc[0] < mean-3*sd): 
            series.drop(index=series.index[0], axis=0, inplace=True)
            n_dropped += 1

    print('Number of observations dropped:', n_dropped)


# Auxiliar function of divide_and_getGeneralFeats
def create_diffs(set_series):
    """Obtains the first differences for the Time Series.
    """
    for series in set_series:
        series['diff'] = abs(series['value'].diff())


# Auxiliar function of divide_and_getGeneralFeats
def divideTS(set_series, window_size, overlap):
    """Joins the stretches obtained by the division of Time Series,
    grouping them by original Time Series.
    """
    series_by_file = []
    for file_series in set_series:
        series_by_file.append(divideTS_fromFile(file_series,
                                                window_size,
                                                overlap))
    return series_by_file


# Auxiliar function of divideTS
def divideTS_fromFile(file_series, window_size, overlap):
    """Divides the Time Series into stretches of length window size
    overlaping or not 50% of the data points.
    """
    
    overlap = 0.5 if overlap else 0

    result = []
    step = int(window_size*(1-overlap))
    stop = False
    start = 0
    
    # Without overlap, if the the division of Time Series creates a last
    # stretch with only one data point, ignore the 1st data point to avoid
    # a single masked data point in the last stretch
    if len(file_series)%window_size == 1 and overlap == 0: 
        start = 1
    
    for low in range(start, len(file_series), step):

        # Obtain the stretch according to the window size and the overlap
        stretch = file_series[low:low+window_size]
            
        if len(file_series) > low+window_size:
            values = np.array(stretch['value'])
            diffs = np.array(stretch['diff'][1:])
        
        # Apply masking in the last stretch if this do not has enough data
        # points to create a window by itself.
        else:
            arr_val = np.append(
                stretch['value'],
                [_ for _ in range(len(file_series), low+window_size)],
                )
            mask = [j >= len(file_series) for j in range(low, low+window_size)]
            values = np.ma.masked_array(arr_val, mask)
                    
            arr_diff = np.append(
                stretch['diff'][1:],
                [_ for _ in range(len(file_series),low+window_size)],
                )
            diffs = np.ma.masked_array(arr_diff, mask[1:])
            
            stop = True
                    
        label = 1 if sum(stretch['is_anomaly']) != 0 else 0
        result.append([values, diffs, label])
                                        
        if stop:
            break
        
    return pd.DataFrame(result).rename(columns={0:'value',
                                                1:'diff',
                                                2:'label'})


# Auxiliar function of divide_and_getGeneralFeats
def all_generalFeats(series_by_file, window_size, overlap, seed, mode):
    """Obtains a pre-selection of features - General Features - for all
    stretches of all original Time Series and returns the shuffled dataframe
    of the results obtained.
    """
    df = pd.DataFrame()
    for file_TS in series_by_file:
        df = pd.concat([df, generalFeats(file_TS, window_size, overlap, mode)])
    df = df.reset_index(drop=True)
    shuffled = df.sample(frac=1, random_state=seed)
    return shuffled


# Auxiliar function of all_generalFeats
def generalFeats(file_TS,window_size,overlap,mode):
    """Obtains, for a certain file, features with regard to the original Time 
    Series from which the strecth was extracted and from the stretch itself
    """
    
    file_info = []

    # Initialize variable to store features obtained from the data points
    # observed before the considered stretch 
    mean_medianBef, mean_medianWind = 0, 0
    sdBefore, sdWind = 0, 0
    
    # Initialize lists containing values of local features for the stretches
    mean_values, sd_values, skew_values = [], [], []

    # If using the Differences the first data point should not be considered
    # as it results from the difference between the fisrt data point of the
    # stretch and a point that do not belongs to the stretch
    diff_correction=1 if mode=='diff' else 0

    if overlap:
        step = int(window_size*0.5)
        list_values = file_TS[mode].iloc[0][:step-diff_correction]
        start, stop = -11, -1  #Last 10 windows
    else:
        list_values = []
        start, stop = -5, None  #Last 5 windows

    for n_index in file_TS.index:

        # Obtain the selected stretch and corresponding label
        value_loc = file_TS[mode][n_index]
        label = file_TS['label'][n_index]

        # Obtain the local features
        meanLoc = np.mean(value_loc)
        sdLoc = np.std(value_loc)
        skewLoc = skew(value_loc,nan_policy='omit')
        medianLoc = np.ma.median(value_loc)

        # Append the newly observed data points of the considered stretch 
        # to the ones previously observed from the original Time Series.
        if overlap:
            list_values = np.ma.concatenate(
                [list_values, file_TS[mode][n_index][step-diff_correction:]]
                )
        else:
            list_values = np.ma.concatenate([list_values,value_loc])
  
        # Obtain features from the set of data points that combines the 
        # considered stretch and data points previously observed.
        if overlap:
            meanAfter, sdAfter,\
                       meanAfterwind = update_after(list_values,
                                                    window_size-diff_correction,
                                                    step, 0.5)
        else:
            meanAfter, sdAfter,\
                       meanAfterwind = update_after(list_values,
                                                    window_size-diff_correction,
                                                    None, 0)

        # Store Features
        file_info.append([label,
                          sdBefore, sdWind,
                          meanAfterwind,
                          mean_medianBef, mean_medianWind,
                          sdLoc, skewLoc,
                          medianLoc, meanLoc])
        
        df_add = pd.DataFrame(file_info,
                          columns=['label',
                                   'sdBefore_'+mode, 'sdWind_'+mode,
                                   'meanAfterwind_'+mode,
                                   'mean_medianBef_'+mode, 'mean_medianWind_'+mode,
                                   'sdLoc_'+mode, 'skewLoc_'+mode,
                                   'medianLoc_'+mode, 'meanLoc_'+mode])

        # Update lists containing values of local features for the stretches
        sd_values = np.append(sd_values, sdLoc)
        skew_values = np.append(skew_values, skewLoc)
        mean_values = np.append(mean_values, meanLoc)

        # Updates features obtained from the data points observed before the
        # considered stretch after the addition of the newly observed stretch
        if overlap:
            sdBefore = np.std(list_values[:-step])
            sdWind = np.std(
                list_values[max(0, len(list_values)-window_size+diff_correction-10*step):-step]
                )
        else:
            sdBefore = sdAfter 
            sdWind = np.std(
                list_values[max(0, len(list_values)-5*(window_size-diff_correction)):]
                )

        # Update summary features from the previously observed stretches
        mean_medianBef = np.ma.median(mean_values)
        if mean_values[start:stop] != []:
            mean_medianWind = np.ma.median(mean_values[start:stop])
        else:
            mean_medianWind = 0

    # Obtain Features from all the data points of the original Time Series
    meanTot = pd.DataFrame([meanAfter]*len(file_info),
                           columns=['meanTot_'+mode])
    sd_meanTot = pd.DataFrame([np.mean(sd_values)]*len(file_info),
                              columns=['sd-meanTot_'+mode])
    sd_maxTot = pd.DataFrame([np.max(sd_values)]*len(file_info),
                             columns=['sd-maxTot_'+mode])
    skew_meanTot = pd.DataFrame([np.mean(skew_values)]*len(file_info),
                                columns=['skew-meanTot_'+mode])
    mean_maxTot = pd.DataFrame([np.max(mean_values)]*len(file_info),
                               columns=['mean-maxTot_'+mode])

    # Store Features from all the data points of the original Time Series
    df_add = pd.concat(
        [df_add, meanTot, sd_meanTot, sd_maxTot, skew_meanTot, mean_maxTot],
        axis=1
        )
        
    return df_add


# Auxiliar function of generalFeats
def update_after(list_values, window_size, step, overlap):
    """Calculates features of the set data points that combines the considered
    stretch and data points previously observed.
    """
    meanAfter = np.mean(list_values)
    sdAfter = np.std(list_values)
    
    if overlap:
        start = max(0, len(list_values)-window_size-10*step)
    else:
        start = max(0, len(list_values)-6*(window_size))
    list_values_afterwind = list_values[start:]
    meanAfterwind = np.mean(list_values_afterwind)

    return meanAfter, sdAfter, meanAfterwind
