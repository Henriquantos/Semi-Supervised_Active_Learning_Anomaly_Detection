from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

def get_bestModel(sfs, interval_of_feats,title):
    """Receives a fitted Sequential Feature Selection objected, 
    a tuple with the tested number of features for the ideal feature vector
    and a string title. 
    Returns the name of the features that belong to the ideal feature vector 
    (in alphabethical order) and the f1_score obtained when the stopping
    criterion is activated. Also returns a figure with the F1-Scores across
    all possibilities of interval_of_feats.
    """
    for i in range(interval_of_feats[0], interval_of_feats[1]):
        if (sfs.subsets_[i+1]['avg_score']-sfs.subsets_[i]['avg_score']) < 0.001:
            break
    
    plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.ylim([0.85, 0.97])
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.savefig(title+'.jpeg')
    return list(sfs.subsets_[i]['feature_names']), sfs.subsets_[i]['avg_score']

