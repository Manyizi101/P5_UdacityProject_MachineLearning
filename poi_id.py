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

# Classifier 1: Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

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
