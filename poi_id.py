#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
# You will need to use more features
features_list = ['poi', "shared_receipt_with_poi",
                 "exercised_stock_options", "bonus"]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
del data_dict["TOTAL"]

# Task 3: Create new feature(s)
# Setting NaNs to 0 and adding new feature
for person in data_dict.keys():
    for k in data_dict[person].keys():
        if data_dict[person][k] == "NaN":
            data_dict[person][k] = 0
    data_dict[person]["payout"] = data_dict[person][
        "bonus"] + data_dict[person]["salary"]
    # data_dict[person]["stock"] = data_dict[person]["exercised_stock_options"] + \
    #     data_dict[person]["loan_advances"] + \
    #     data_dict[person]["long_term_incentive"]
    # data_dict[person]["poi_conversation"] = data_dict[person]["from_poi_to_this_person"] + \
    #     data_dict[person]["from_this_person_to_poi"] + \
    #     data_dict[person]["shared_receipt_with_poi"]

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html


# Provided to give you a starting point. Try a variety of classifiers.

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

scaler = MinMaxScaler(feature_range=(0, 1))
nb = DecisionTreeClassifier(criterion="gini")

clf = Pipeline([
    ("scale_features", scaler),
    ("classifier", nb)
])

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
