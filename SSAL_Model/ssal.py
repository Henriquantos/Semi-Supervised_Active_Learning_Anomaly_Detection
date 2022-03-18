import time
import pandas as pd
from sklearn import metrics

from query_strategy import*
from self_train import *


def ssal(x_labeled, y_labeled, x_unlabeled, real_labels, holdout_feats,
         holdout_labels, limit, classifier, seed, qs, st):
    """Applies a Semi Supervised Active Learning Model to classify observations
    when there is a scarcity of labels.
    """
    start_t = time.time()
    num_expert = 0
    expert_consecutive = 0
    keep_going = True
    result = pd.DataFrame(columns=['precision', 'recall',
                                   'f1_score', 'time_spent',
                                   'num_expert', 'for classifying',
                                   'classified'])
    while len(x_unlabeled) != 0 and keep_going:
        # Automatically classification of unlabeled observations
        x_labeled, y_labeled, x_unlabeled, predictions,\
            changes, classifier = self_train(x_labeled, y_labeled, 
                                             x_unlabeled, limit, 
                                             classifier, st)
        # Re-initialize counter of loops without automatically classified
        # observations
        if changes:
            expert_consecutive = 0
        # Obtain predictions and metrics  for the holdout set with the
        # classifier trained with the enlarged set
        preds = classifier.predict(holdout_feats)
        precision=metrics.precision_score(holdout_labels,preds)
        recall=metrics.recall_score(holdout_labels,preds)
        f1_score=2*precision*recall/(precision+recall)
        #Calculate temporal cost until the moment
        last_t = time.time()
        time_spent=last_t-start_t
        # Store metrics obtained
        result = result.append(pd.DataFrame([[precision, recall,
                                              f1_score, time_spent,
                                              num_expert, len(x_unlabeled),
                                              len(x_labeled)]],
                                            columns=['precision', 'recall',
                                                     'f1_score', 'time_spent',
                                                     'num_expert', 'for classifying',
                                                     'classified']
                                            )
                               )        
        if len(x_unlabeled) != 0:
            # Label the most informative observation according to the Query Strategy
            index = QueryStrategy().selection(qs,
                                              x_unlabeled,
                                              seed,
                                              predictions,
                                              y_labeled)
            
            x_labeled = x_labeled.append(x_unlabeled.loc[index])
            y_labeled = y_labeled.append(real_labels.loc[index])
            x_unlabeled = x_unlabeled.drop(index)
            num_expert += 1
            expert_consecutive += 1
        # Stopping Criterion Activation
        if (len(x_labeled)/(len(x_labeled)+len(x_unlabeled)) > 0.5 and expert_consecutive > 5):
            keep_going=False
            
    return result