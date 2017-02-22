# Building the Classifier

## Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question.

The goal of this project is to use a machine learning algorithm to classify former Enron employees as either a "Person of Interest" or not based on similar financial and email profiles amongst the Enron employees in our dataset.

Our dataset is based on the declassified email corpus and some financial documents that were released after the Enron scandal in the early 2000s. The prepared data we were handed has 145 former employees, counts of emails between them and other persons of interest, as well as some salary and stock option information. Using these features, we should be able to systematically identify who was a person worth investigating.

The dataset itself has 146 rows, 145 employees and 1 TOTAL entry. 18 of these 145 employees are actual POI while the remaining 127 are not.

Columns include information on bonuses, deferred payments, director_fees, stock options, restricted_stock, salary, and message exchange counts amongst other POIs.

There are a lot of "NaN" values included but according to the source, they are not invalid entries, but a marker of absence of data. You will notice in my preparation I have filled these values with 0s where appropriate (i.e. filled all NaNs with 0 aside from email field)

### Were there any outliers in the data when you got it, and how did you handle those? [relevant rubric items: "data exploration", "outlier investigation"]

There were some outliers but I figured having unusually high bonus or exercised stock options might be useful in our classification task so if the outliers were valid, I left them.

I did remove one row however, TOTAL. It looks like a sum of all our vales snuck it's way into the data so it had to be removed.

### What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not?

I used a custom features, payout, exercised_stock_options, and from_this_person_to_poi.

Payout is a combination of an employees salary and bonus features.

I tried to make several different features during the course of building my classifier but they were not often selected by my SelectKBest estimator or my SelectFromModel attempts with my LinearRegression and SVC models.

My feature selection process started by plotting all the features against eachother in a scatter plot in Tableau and looking for distinct POI and non POI clusters.

I created a few classifiers, split my data into training and test sets, and looked at the performance in terms of accuracy, precision, and recall. None of these experiments really exceeded the course expectations so I decided to use SelectKBest and SelectFromModel where appropriate. I played around making my own features and eventually settled on the 3 I mentioned above.

I did end up scaling my features using a MinMaxScaler as we have a lot of numeric values but the ranges are extreme. It was much easier to visually compare different features in Tableau once they had been scaled.

### Explain what features you tried to make, and the rationale behind it. In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.

Since our data has two groupings, financial and conversations, I thought I would try and make features that combine the best features in those categories.

I identified which features to use from each section by using SelectKBest where k=7\. I would them look at the 7 strongest features, and lump them together.

I combined salary and bonus in payout, several of our stock options in stock, and our message to and from poi fields in poi_conversation.

I did not end up using any of my features as the performance was stronger without them.

Decision Tree Feature importances:

- Bonus : 0.20287093
- exercised_stock_options: 0.38827463
- shared_receipt_with_poi: 0.40885444

SelectKBest scores:

- Bonus : 2.4830
- exercised_stock_options: 3.6756
- shared_receipt_with_poi: 4.8911

I chose k=3 over k=5 or k=7 as our accuracy, precision, and recall were at their highest with these three features.

### What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?

I ended up using the DecisionTreeClassifier although I tried RandomForest, Naive Bayes (GaussianNB), SVC, and even LogisticRegression.

Accuracy was comparable across the board, but my DecisionTreeClassifier had the best combination of precision and recall scores with RandomForest a close second.

### What does it mean to tune the parameters of an algorithm, and what can happen if you don't do this well? How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier)

Tuning parameters for an algorithm means selecting the best parameters for your algorithm to perform given the condition of your data. In the case of an algorithm like KNearestNeighbors, it is choosing the number of neighbors to collect votes from.

In our example with the DecisionTreeClassifier, it could mean choosing the quality of a split using either Gini impurity or based on information gain (entropy). For this project, this is the only parameter I changed; I let GridSearchCV make the decision for me and it selected 'gini' for my criterion parameter.

Each algorithm is different and have different parameters that you can change to better meet the learning task; this is know as tuning.

If you do not perform this task well, you can have an underfit or overfit model. An underfit model is one that does not see the trend that you are attempting to "learn" while an overfit model is one that is overly tuned that it is too influenced by variations (noise) in our data.

### What is validation, and what's a classic mistake you can make if you do it wrong? How did you validate your analysis?

Validation is a process by which you evaluate how well your model has been trained. Since we have supervised learning project, we can split our data into two sets, a training and a test set.

I changed our splits to be 25-75 test-train.

Our model will have 75% of our dataset to "understand" which rows have which labels. After this training is complete, we feed our model the label-less rows in our test set and compare its predictions against the actual labels in the label test set.

A classic validation mistake is leaving the label in the x or "row" set when testing, resulting in the model having near perfect accuracy.

I formed my test-train split using the train_test_split module from sklearn.cross_validation and sklearn.metrics.precision_recall_fscore_support to get precision, recall, and F1 scores.

### Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm's performance.

Here are two common evaluation metrics and my model's performance:

- Precision: 0.35979
- Recall: 0.37850

Precision is the number of relevant items out of the total items selected. Let's assume Enron has 1,000 employees, we try to identify POIs from this group and we come up with a list of 100\. Of those 100, only 35 are actually guilty. In this example we would have a precision of 35/100.

Recall is the number of relevant items from what we selected. Let's say we are trying to determine the number of toucans in a rainforest. We spot 30 beautiful toucans and conclude that there are only 30 toucans in this particular rainforest. A study later reveals there were actually 33 birds, we found 30 of them but 3 managed to elude us! Therefore our recall is 30/33.

For this project this translates to: Precision - 0.35979 of the people (roughly 52 people) I identified as POIs, were actually POIs.

Recall - 0.37850, of those that I selected, I only captured this proportion of the total POIs. I.E. If there were 18 POIs, I would have only identified 7 people.
