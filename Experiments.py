
# coding: utf-8

# In[154]:

import pickle
import pandas as pd
import numpy as np

# importing our classification steps
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support 

# importing our models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[262]:

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

del data_dict["TOTAL"]

# transpose rows and columns
df = pd.DataFrame(data_dict).transpose()

# list of columns we want as numeric
numeric_columns = ["bonus", "deferral_payments", "deferred_income", "director_fees",   "exercised_stock_options", "expenses", "from_messages", "from_poi_to_this_person",   "from_this_person_to_poi", "loan_advances", "long_term_incentive", "other", "restricted_stock",    "restricted_stock_deferred", "salary", "shared_receipt_with_poi", "to_messages",    "total_payments", "total_stock_value"]

df[numeric_columns] = df[numeric_columns].apply(lambda x: pd.to_numeric(x, errors="coerce"))

# imputation
df.fillna(0, inplace=True)

# creating our new features
# df["payout"] = df["bonus"] + df["salary"]+df["total_stock_value"]
# df["stock"] = df["exercised_stock_options"] + df["loan_advances"] + df["long_term_incentive"]
# df["poi_conversation"] = df["from_poi_to_this_person"] + df["shared_receipt_with_poi"]

# Changing our poi field from object to int so it appears in our correlation table
labels = np.where(df["poi"] == True, 1, 0)

# Drop unnecessary columns
del df["email_address"]
del df["total_payments"]
del df["total_stock_value"]
del df["loan_advances"]

columns = df.columns

# Scale features
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=columns)

df.to_csv("EnronData.csv")

# Creating our training data
df_train = df[list(df.columns)]
df_label = df_train["poi"]
del df_train["poi"]

x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, test_size=0.25)


# In[13]:

def custom_score(clf, x, y):
    accuracy = clf.score(x, y)
    y_true = y
    y_pred = clf.predict(x)
    return  precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=[0,1], average="macro")[:-1]


# In[14]:

def get_features_used(data, k_best_estimator):
    return list(df_train.columns[k_best.get_support()])


# In[15]:

def report_performance(clf, x, y, data, k_best):
    features = get_features_used(data, k_best)
    accuracy = clf.score(x,y)
    precision, recall, f1 = custom_score(clf, x, y)
    return "Features used: {}\nAccuracy: {}\nPrecision: {}\nRecall: {}\nF1: {}".format(features, accuracy, precision, recall, f1)


# ### Decision Tree

# In[265]:

dtc = DecisionTreeClassifier()

k_best = SelectKBest(score_func=chi2, k=3)

clf_param = {"criterion": ("gini", "entropy")}
clf_grid = GridSearchCV(dtc, clf_param)

clf = Pipeline([
        ("feature_selection", k_best),
        ("classifier", clf_grid)
    ])

clf.fit(x_train, y_train)
print report_performance(clf, x_test, y_test, df_train, k_best)


# In[267]:

list(k_best.scores_)


# ### Random Forest

# In[236]:

rfc = RandomForestClassifier()

k_best = SelectKBest(score_func=chi2, k=3)

clf_param = {"n_estimators": [1, 10, 100, 1000], "criterion": ("gini", "entropy")}
clf_grid = GridSearchCV(rfc, clf_param)

clf = Pipeline([
        ("feature_selection", k_best),
        ("classifier", clf_grid)
    ])

clf.fit(x_train, y_train)
print report_performance(clf, x_test, y_test, df_train, k_best)


# ### Naive Bayes

# In[237]:

nb = GaussianNB()

k_best = SelectKBest(score_func=chi2, k=3)

clf = Pipeline([
        ("feature_selection", k_best),
        ("classifier", nb)
    ])

clf.fit(x_train, y_train)
print report_performance(clf, x_test, y_test, df_train, k_best)


# ### SVM SVC

# In[238]:

svc = SVC()

k_best = SelectKBest(score_func=chi2, k=3)

clf_param = {"kernel": ("linear", "rbf"), "C": [1, 10, 100, 1000]}
clf_grid = GridSearchCV(svc, clf_param)

clf = Pipeline([
        ("feature_selection", k_best),
        ("classifier", clf_grid)
    ])

clf.fit(x_train, y_train)
print report_performance(clf, x_test, y_test, df_train, k_best)


# ### Logistic Regression

# In[239]:

lr = LogisticRegression()

k_best = SelectKBest(score_func=chi2, k=3)

clf_param = {"C": [1, 10, 100, 1000]}
clf_grid = GridSearchCV(lr, clf_param)

clf = Pipeline([
        ("feature_selection", k_best),
        ("classifier", clf_grid)
    ])

clf.fit(x_train, y_train)
print report_performance(clf, x_test, y_test, df_train, k_best)


# In[ ]:



