import random
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd


class QueryStrategy:

    def selection(self, qs, x_unlabeled, seed, predictions, y_labeled):
        if qs=='R':
            return self.R_selection(x_unlabeled, seed)
        elif qs=='Unc':
            return self.Unc_selection(predictions)
        elif qs=='Ut':
            return self.Ut_selection(x_unlabeled, x_unlabeled)
        elif qs=='Unc&Ut1':
            return self.UncUt1_selection(predictions, x_unlabeled)
        elif qs=='Unc&Ut2':
            return self.UncUt2_selection(predictions, x_unlabeled)
        elif qs=='Unc&Ut1_variation':
            return self.variation(QueryStrategy.UncUt1_selection,
                                  predictions,
                                  x_unlabeled,
                                  y_labeled)
        elif qs=='Unc&Ut2_variation':
            return self.variation(QueryStrategy.UncUt2_selection,
                                  predictions,
                                  x_unlabeled,
                                  y_labeled)
        else:
            print('The passed Query Strategy does not exist')


    def R_selection(self, x_unlabeled, seed):
        random.seed(seed)
        index = random.sample(list(x_unlabeled.index), 1)
        return index


    def Unc_selection(self, predictions):
        uncertainty = dict()
        # Obtain the ordered uncertainty of all observations
        for pred in predictions.items():
            uncertainty[pred[0]] = abs(pred[1][0]-pred[1][1])
        sorted_uncertainty = {
            k: v for k,v in sorted(uncertainty.items(),key=lambda item:item[1])
            }
        # Select the most uncertain observation
        index = list(sorted_uncertainty.keys())[0]
        return index


    def Ut_selection(self, unlabeled_selected,x_unlabeled):
        # Obtain the utility for the selected unlabeled observations
        dist = euclidean_distances(unlabeled_selected, x_unlabeled)
        dist_df = pd.DataFrame(dist,
                               columns=[x_unlabeled.index],
                               index=[unlabeled_selected.index])  
        # Attribute a high distance from the observation to itself
        for col in dist_df:
            for row in dist_df.index:
                if col == row:
                    dist_df[col][row] = 10000
        # Select the most useful observation
        index = dist_df.idxmin().value_counts().idxmax()[0]
        return index


    def UncUt1_selection(self, predictions, x_unlabeled):
        uncertainty = dict()
        # Obtain the ordered uncertainty of all observations
        for pred in predictions.items():
            uncertainty[pred[0]] = abs(pred[1][0]-pred[1][1])
        sorted_uncertainty={
            k: v for k,v in sorted(uncertainty.items(),key=lambda item:item[1])
            }
        # Select the all the observation with higher value of uncertainty
        uncertainty_values = list(sorted_uncertainty.values())
        until = uncertainty_values.count(uncertainty_values[0])
        most_uncertain=dict(list(sorted_uncertainty.items())[:until])
        # Obtain the utility for the most uncertain observations
        dist = euclidean_distances(x_unlabeled.loc[most_uncertain.keys()],
                                   x_unlabeled)
        dist_df = pd.DataFrame(dist,
                               columns=[x_unlabeled.index],
                               index=[x_unlabeled.loc[most_uncertain.keys()].index])
        # Attribute a high distance from the observation to itself
        for col in dist_df:
            for row in dist_df.index:
                if col == row:
                    dist_df[col][row] = 10000
        # Select the most informative observation
        index = dist_df.idxmin().value_counts().idxmax()[0]
        return index



    def UncUt2_selection(self, predictions, x_unlabeled):
        uncertainty = dict()
        # Obtain the uncertainty of all observations
        for pred in predictions.items():
            uncertainty[pred[0]] = abs(pred[1][0]-pred[1][1])
        sorted_uncertainty={
            k: v for k,v in sorted(uncertainty.items(),key=lambda item:item[1])
            }
        # Select the 20% most uncertain observations
        uncertainty_values = list(sorted_uncertainty.values())
        unc1 = uncertainty_values.count(uncertainty_values[0])
        unc2 = 0
        n_unlabeled = len(uncertainty)
        while (unc1+unc2)/n_unlabeled < 0.2:
            unc2 = uncertainty_values.count(uncertainty_values[unc1])
            unc1 += unc2
        most_uncertain = dict(list(sorted_uncertainty.items())[:unc1])
        # Obtain the utility for the most uncertain observations
        dist = euclidean_distances(x_unlabeled.loc[most_uncertain.keys()],
                                   x_unlabeled)
        dist_df = pd.DataFrame(
            dist,
            columns=[x_unlabeled.index],
            index=[x_unlabeled.loc[most_uncertain.keys()].index],
            )
        # Attribute a high distance from the observation to itself
        for col in dist_df:
            for row in dist_df.index:
                if col == row:
                    dist_df[col][row] = 10000
        utility = dist_df.idxmin().value_counts()
        # Calculate informativeness based on uncertainty and utility
        select_dict={}
        for index in utility.index:
            select_dict[index[0]] = (0.5*most_uncertain[index[0]]
                                    + 0.5*(1-utility[index]/len(x_unlabeled)))
        sorted_select={
            k: v for k,v in sorted(select_dict.items(), key=lambda item:item[1])
            }
        # Select the most informative observation
        index=list(sorted_select.keys())[0]
        return index


    def variation(self, QS, predictions, x_unlabeled, y_labeled):
        # Obtain maximal & minimal probabilities of an observation being normal
        aux = {k: v[0] for k, v in predictions.items()}
        minimum = min(aux.values())
        maximum=max(aux.values())
        # If % anomalies is less than 5% try to find anomaly
        if sum(y_labeled['label'])/len(y_labeled) < 0.05: 
            # If  minimal probability of being normal is higher than 50%,
            # there are no anomalies according to the trained classifier. 
            # Despite that, select the observations with higher probability
            # of being anomalies, i.e, lower probability of being normal.
            # Call Utility Selection Query Strategy on those observations.
            if minimum >= 0.5:
                chosen_preds = {
                    k: v for k, v in predictions.items() if v[0] == min(aux.values())
                    }
                unlabeled_selected = x_unlabeled.loc[list(chosen_preds.keys())]
                return self.Ut_selection(unlabeled_selected, x_unlabeled)
            # If there are anomalies according to the classifier call the
            # Query Strategy passed to the function
            else:
                chosen_preds = {
                    k:v for k,v in predictions.items() if v[0] < 0.5
                    }
                return QS(self, chosen_preds, x_unlabeled)
        # If % anomalies is more than 15% try to find normal observation
        elif sum(y_labeled['label'])/len(y_labeled) > 0.15:
            # If  maximal probability of being normal is lower than 50%,
            # there are no normal observations according to the trained 
            # classifier. Despite that, select the observations with 
            # higher probability of being normal and call Utility Selection 
            # Query Strategy on those observations.
            if maximum <= 0.5:
                chosen_preds = {k:v for k, v in predictions.items() if v[0] == max(aux.values())}
                unlabeled_selected = x_unlabeled.loc[list(chosen_preds.keys())]
                return self.Ut_selection(unlabeled_selected, x_unlabeled)
            # If there are normal observations according to the classifier
            # call the Query Strategy passed to the function
            else:
                chosen_preds = {
                    k:v for k,v in predictions.items() if v[0] > 0.5
                    }
                return QS(self, chosen_preds, x_unlabeled)
        # If % anomalies between 5% and 15% call the Query Strategy passed to
        # the function without applying any filtering
        else:
            chosen_preds = predictions
            return QS(self, chosen_preds, x_unlabeled)
