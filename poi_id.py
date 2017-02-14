#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas
sys.path.append("../tools/")

from time import time
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Attempt 1: Using all features
features_list = ['poi','bonus', 'deferral_payments', 'deferred_income',\
    'director_fees', 'exercised_stock_options', 'expenses', 'from_messages',
    'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances',
    'long_term_incentive', 'other', 'restricted_stock',
    'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
    'to_messages', 'total_payments', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###############################################################################
### Task 2: Remove outliers
### Remove TOTAL row from dataset
data_dict.pop('TOTAL')

### Remove THE TRAVEL AGENCY IN THE PARK row from dataset
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

###############################################################################
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

## Converting dictionary to pandas data frame
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# Set the index of df to be the employees series
df.set_index(employees, inplace=True)

df.replace(to_replace='NaN', value=0, inplace=True)

# Add new feature
df['ratio_email_sent_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['ratio_email_from_poi'] = df['from_poi_to_this_person']/df['to_messages']
df.replace(to_replace='NaN', value=0, inplace=True)

# Create a new list of features
new_features_list = df.columns.values
print new_features_list

# Convert data frame back to dictionary
my_dataset = df.to_dict('index')


###############################################################################
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Split dataset to training set and testing set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

###############################################################################
### Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif
skb = SelectKBest(f_classif, k = 'all')
skb.fit(features_train, labels_train)

features_selected = [features_list[i+1] for i in skb.get_support(indices = True)]
print 'The features selected by Select K Best are: '
print features_selected

features_selected.insert(0, 'poi')
print features_selected

# Features selected are: 'bonus', 'exercised_stock_options', 'salary',
#                        'shared_receipt_with_poi', 'total_stock_value'
# The same features were suggested by our exploratory data analysis

###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## PCA
# n_components = 5

# print "Extracting the top %d eigenvalues from %d features" % (n_components, len(features_train))
# t0 = time()
# pca = PCA(n_components = n_components, whiten = True).fit(features_train)
# print "done in %0.3fs" % (time() - t0)

# print "Projecting the input data on the eigenvalue orthonormal basis"
# t0 = time()
# features_train_pca = pca.transform(features_train)
# features_test_pca = pca.transform(features_test)
# print "done in %0.3fs" % (time() - t0)

## Classifier 1: Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
# clf = clf.fit(features_train_pca, labels_train)
# pred = clf.predict(features_test_pca)

## Classifier 2: Support Vector Machine (assume balanced data)
# from sklearn.svm import SVC
# clf = SVC(C = 1000, kernel = 'sigmoid', gamma = 0.1)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 3: Decision Tree
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 4: K-Nearest Neighbours
#from sklearn import neighbors
#clf = neighbors.KNeighborsClassifier(weights = 'distance')
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 5: AdaBoost
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10), \
#                          random_state=42)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 6: Random forest
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, \
#                                random_state=42)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

min_max_scaler = MinMaxScaler()
pca = PCA()
pipeline = Pipeline(steps = [('scaler', min_max_scaler), ('pca', pca), ('classifier', clf)])
print pipeline.get_params().keys()

param_pca = {'pca__n_components': [3,5,7,9,10]}
cv = StratifiedShuffleSplit(n_splits = 100, test_size = 0.3, random_state=42)

gs = GridSearchCV(pipeline, param_grid = param_pca, cv=cv, scoring = 'f1')
gs.fit(features, labels)
clf = gs.best_estimator_

###############################################################################

## Compute accuracy score of classifier
## Result: Naive Bayes = 0.88
## Comment: Accuracy score is not a good evaluation method for this dataset
##          because the majority of employees are non-POI (77 out of 95).
##          Hence, guessing that someone is a non-POI will give a high accuracy.
##          A better evaluation method is probably precision-recall score.
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(pred, labels_test)
# print "Classifier 1 (Naive Bayes) Accuracy: ", acc

## Generate confusion matrix
#from sklearn.metrics import confusion_matrix
#print confusion_matrix(labels_test, pred)

## Generate classification report
#from sklearn.metrics import classification_report
#print classification_report(labels_test, pred)

## Classifier 1: Naive Bayes
## Result: Precision-recall of POI = 0.5, 0.4
## Comment: Low precision and recall. Naive Bayes has a strong assumption
##          that all features are independent of each other.
## Follow-up: Conduct PCA before running Naive Bayes to ensure features are
##            independent of each other
## PCA Result: Precision-recall of POI = 0.6, 0.6 (n_components = 5) or
##                                       0.67, 0.4 (n_components = 3)
## Comment: As expected, Naive Bayes performs better after PCA

## Classifier 2: Support Vector Machine (assume balanced data)
## Result: Precision-recall of POI = 0, 0
## Comment: Very low precision and recall. Data is highly unbalanced
##          - there are a lot more non-POIs than POIs. Hyperplane from SVM
##          relies much on non-POI points, resulting in inaccurate separation
## Follow-up: Perhaps SVM would perform better if we treat the data imbalance
## http://stackoverflow.com/questions/18078084/how-should-i-teach-machine-learning-algorithm-using-data-with-big-disproportion
## https://discussions.udacity.com/t/question-about-feature-selection-and-classifier-performance/166278/2

## Classifier 3: Decision Tree (assume balanced data, min_samples_split = 15)
## Result: Precision-recall of POI = 0.33, 0.4
## Comment: Low precision and recall. Better results with higher min_samples_split
##          because tree doesn't overfit training set. However, data is highly
##          unbalanced, resulting in a biased tree.
## Follow-up: Need to balance dataset before running decision tree.

## Classifier 4: K-Nearest Neighbors (k = 3)
## Result: Precision-recall of POI = 0.5, 0.4
## Comment: Low precision and recall.
##          k = 3 performs better than k = 1. Since more neighbors are
##          considered for classification, the result is less biased.
##          KNN is known to be computationally expensive. However, since Enron
##          dataset is not that large, speed is not a problem.
## Follow-up: KNN is also known to be a lazy learner because it does not learn
##            from training dataset. It might not generalize well for other
##            testing sets. Cross validation is needed to evaluate KNN better.
## GridSearch result: KNN performs very poorly upon cross validation

## Classifier 5: Adaboost (decision tree, min_samples_split = 10, n_estimators = 200)
## Result: Precision-recall of POI = 0.5, 0.2
## Comment: Low precision, very low recall. Low recall might be because
##          decision tree is unable to classify unbalanced data well
## Follow-up: Need to balance dataset before running adaboost

## Classifier 6: Random forest (min_samples_split = 5, n_estimators = 100)
## Result: Precision_recall of POI = 1.0, 0.2
## Comment: High precision, very low recall. Low recall might be because
##          decision tree is unable to classify unbalanced data well.
## Follow-up: Need to balance dataset before running random forest


###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from tester import test_classifier
print "Tester Classification Report"
test_classifier(clf, my_dataset, features_list, folds = 1000)

###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
