# Semi-Supervised_Active_Learning_Anomaly_Detection

(1) DATASET
The developed model uses the Yahoo! Synthetic and real time-series with labelled
anomalies, version 1.0 dataset, available at 
https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70

The dataset is composed by 367 files and each file represents one Time Series
of real or synthetically created data. The files are divided into 4 folders:
A1Benchmark/real_(int).csv - with int between 1 and 67
A2Benchmark/synthetic_(int).csv - with int between 1 and 100
A3Benchmark/A3Benchmark-TS(int).csv - with int between 1 and 100
A4Benchmark/A4Benchmark-TS(int).csv - with int between 1 and 100

The real data 'is based on real production traffic to some of the Yahoo!
properties'. The Time Series have variable scale and length. All of them
have a timestamp, a value (metric observed) and a label.

In A1Benchmark the timestamp is a range from 0 to length of the time series.
In A[2-4]Benchmark timestamp is an UNIX timestamp.
For both cases, each data point represents one hour worth of data.

The label is named 'is_anomaly' for A[1-2]Benchmark and 'anomaly' for
A[3-4]Benchmark. For both cases it takes value 1 if the observation the
data point is anomalous and 0 otherwise.

(2) Ideal Feature Vector Selection
In this folder is presented a method to select the ideal feature vector.
Firstly, the Time Series are divided using a pre-specified window size
and using or not an overlap of 50%.
In the file 'main_Automatic_FeatureVector_Selection.py' the features are
extracted from the Time Series Feature Extraction Library (TSFEL).
In the file 'main_Ideal_FeatureVector_Selection.py' the selection is
performed from a pool of 162 features, being 156 of them interval based
and small variations from each other to maximize the discrimanating power.
This pool can either be extrated from the Original Values of the Time Series,
from the first differences of it or from both. After extracting the 162
features, the highly correlated ones are dropped.
For both cases it is applied a Sequential Feature Selection using the MLXtend
library to select the subset of features that yields a higher F1-Score in a
10-fold Cross-Validation with the range for the number of features wanted
previously specified.
The remaining files contain the auxiliar functions applied in the two
previously described files.

(3) SSAL Model
To build the model, we will use a subset of 11 features from the initial
pool of 162, and the Random Forest Classifier with 20 decision trees as
this ensemble yielded the higher F1-Score of 94.82%. To achieve this 
F1-Score it was used a window size of 24 to represent one day worth of
data, an overlap of 50% to avoid the the cutting of the signal in
inconvinient locations and the features were extracted from the
Differences obtained.
The objective of the model is to be capable of classifiying unlabeled
observations when the the quantity of labels is scarce. To accomplish
it, the model is composed by:
 -> a Semi-Supervised Learning segment in which a Self-Training algorithm
is applied. The classification of unlabeled observations performed
by this algorithm can either be Non-Recursive or Recursive, being the
Self-Training ran a single time per iteration or iteratively until a
stopping criterion is met, respectively.
 -> an Active Learning segment in which a Query Strategy selects an
observation to be labeled by the expert. The consulting of the expert
is simulated by the querying of the hidden label of the selected
observation. The tested Query Strategies are:
     -- Random-based
     -- Uncertainty based
     -- Utility based
     -- Uncertainty&Utility1 based -> between the observations
     with higher uncertainty value selects the msot useful
     -- Uncertainty&Utility2 based -> for the 20% of higher
     uncertainty observations calculates the utility and then performs
     a linear combination to determine the most informative
Additionaly, a filter can be applied to the Uncertainty&Utility
based Query Strategies that allows to increase the probability of
selecting an abnormal observations when the percentage of anomalies
in the labeled set is lower than expected or a normal observation
in the opposite situation.

The model classification is stopped when more than 50% of the training
dataset is classified and the expert is 5 or more times consulted
without resulting in any automatically classified observations of the
training set during these loops.


