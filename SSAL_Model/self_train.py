import pandas as pd

def self_train(x_labeled, y_labeled, x_unlabeled, limit, classifier, st):
    """Automatically classifies the unlabeled observations for which the
    classifier has a certainty above limit. If st is 'nrl' only one iteration
    is performed and if it is 'rl' multiple loops are performed until the
    stopping criterion is activated.
    """
    stop = False
    classifier = classifier
    number_loops = 1
    while not stop:
        classifier.fit(x_labeled, pd.Series.ravel(y_labeled))
        predictions = dict(zip(list(x_unlabeled.index),
                               classifier.predict_proba(x_unlabeled))
                           )
        num_changes = 0
        for pred in predictions.items():
            index = pred[0]
            # Probability being normal higher than limit -> attribute label 0
            if pred[1][0] >= limit:
                x_labeled = x_labeled.append(x_unlabeled.loc[index])
                y_labeled = y_labeled.append(pd.DataFrame([0],
                                                          index=[index],
                                                          columns=['label']))
                x_unlabeled = x_unlabeled.drop(index) 
                num_changes += 1
            # Probability being anomaly higher than limit -> attribute label 1
            elif pred[1][1] >= limit:
                x_labeled = x_labeled.append(x_unlabeled.loc[index])
                y_labeled = y_labeled.append(pd.DataFrame([1],
                                                          index=[index],
                                                          columns=['label']))
                x_unlabeled = x_unlabeled.drop(index) 
                num_changes += 1
        # If there were no changes or if all the unlabelled set was classified
        # the Stopping Criterion of the Self-Training is activated
        if num_changes == 0 or len(x_unlabeled) == 0:
            if number_loops==1:
                changes=False
            stop=True
        # If using Non-Recursive Labelling stop after one iteration
        elif st == 'nrl':
            changes=True
            stop=True
        else:
            assert st == 'rl'
            number_loops += 1
            changes = True
    return x_labeled, y_labeled, x_unlabeled, predictions, changes, classifier