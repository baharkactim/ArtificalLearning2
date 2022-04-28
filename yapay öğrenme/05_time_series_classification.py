# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:08:26 2021
@author: acseckin
"""

from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# This is a preprocessed time series data. 
# https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
filepath="data/UCI HAR Dataset/"
# load the dataset, returns train and test X and y elements
trainX_raw=read_csv(filepath + "train/X_train.txt", header=None, delim_whitespace=True)
trainy_raw=read_csv(filepath + "train/y_train.txt", header=None, delim_whitespace=True)
trainX=trainX_raw.values
trainy=trainy_raw.values
print("Train X shape:",trainX.shape, "Train y shape:", trainy.shape)

testX_raw=read_csv(filepath + "test/X_test.txt", header=None, delim_whitespace=True)
testy_raw=read_csv(filepath + "test/y_test.txt", header=None, delim_whitespace=True)
testX=testX_raw.values
testy=testy_raw.values
print("Test X shape:",testX.shape, "Test y shape:",testy.shape)

trainy, testy = trainy[:,0], testy[:,0]

models=dict()
models['KNN'] = KNeighborsClassifier(n_neighbors=7)
models['DT'] = DecisionTreeClassifier()
models['SVC'] = SVC()
models['GNB'] = GaussianNB()
# ensemble models
models['BAG'] = BaggingClassifier(n_estimators=100)
models['RF'] = RandomForestClassifier(n_estimators=100)
models['ExT'] = ExtraTreesClassifier(n_estimators=100)
models['GBC'] = GradientBoostingClassifier(n_estimators=100)
print('Numbers of models: %d' % len(models))

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('%s, Classification Score=%.3f' % (name, score))

# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)