#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

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

### Task 2: Remove outliers
### Remove TOTAL row from dataset
data_dict.pop('TOTAL')


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Split dataset to training set and testing set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## Classifier 1: Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 2: Support Vector Machine (assume balanced data)
# from sklearn.svm import SVC
# clf = SVC(C = 1000, kernel = 'sigmoid', gamma = 0.1)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 3: Decision Tree
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(min_samples_split = 15)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 4: K-Nearest Neighbours
# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 5: AdaBoost
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10), \
#                            n_estimators = 200, random_state=42)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

## Classifier 6: Random forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=100, min_samples_split=5, \
#                                 random_state=42)
# clf = clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)

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
from sklearn.metrics import confusion_matrix
print confusion_matrix(labels_test, pred)

## Generate classification report
## Classifier 1: Naive Bayes
## Result: Precision-recall of POI = 0.5, 0.4
## Comment: Low precision and recall. Naive Bayes has a strong assumption
##          that all features are independent of each other.
## Follow-up: Conduct PCA before running Naive Bayes to ensure features are
##            independent of each other

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

from sklearn.metrics import classification_report
print classification_report(labels_test, pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
