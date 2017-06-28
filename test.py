import numpy as np
import pandas as pd
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
try: # depends on the version of scikit-learn installed
	from sklearn.model_selection import cross_val_score
except:
	from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

def parse_argument():
	"""
	Code for parsing arguments
	"""
	parser = argparse.ArgumentParser(description='Parsing a file.')
	parser.add_argument('--train', nargs=1, required=True)
	parser.add_argument('--test', nargs=1, required=True)
	args = vars(parser.parse_args())
	return args


def parse_train_data(filename, ycol_name='default'):
	""" Given a filename return X and Y numpy arrays
	X is of size number of rows x num_features
	Y is an array of size the number of rows
	Y is the second element of each row.
	"""
	df = pd.read_csv(filename, sep=",", header=0)
	df = df.dropna(subset = [ycol_name])
	X = df.drop(ycol_name, axis=1).values
	Y = df[ycol_name].values
	return X, Y


def parse_test_data(filename):
	""" Given a filename return X numpy array
	X is of size number of rows x num_features
	"""
	df = pd.read_csv(filename, sep=",", header=0)
	X = df.values
	return X


def new_label(Y):
	""" Transforms a vector of `True`s and `False`s in `1`s and `-1`s.
	"""
	return [1. if y==True else -1. for y in Y]


def old_label(Y):
	return [True if y == 1. else False for y in Y]


def accuracy(y, pred):
	return np.sum([y[i]==pred[i] for i in range(len(y))])/float(len(y))
	#return np.sum(y == pred) / float(len(y)) 


def writeResults(x, pred):
	with open("predictions.csv", "w") as outfile:
		for i in range(len(y)):
			outfile.write("{},{}".format(x[i][0], pred[i])+os.linesep)


def main():
	"""
	This code is called from the command line via
	python nubank.py --train [path to filename] --test [path to filename] --numTrees [int]
	"""
	args = parse_argument()
	train_file = args['train'][0]
	test_file = args['test'][0]
	num_trees = 50#int(args['numTrees'][0])

	# read data as numpy arrays
	X_train, Y_train = parse_train_data(train_file)
	X_test = parse_test_data(test_file)
	Y_train = new_label(Y_train)

	estimator = Pipeline([("imputer", Imputer(missing_values="NaN",
		                                      strategy="mean",
		                                      axis=0)),
	                      ("forest", RandomForestClassifier(n_estimators=100))])
	estimator.fit(X_train, Y_train)
	Yhat = estimator.predict(X_train)
	Yhat_test = estimator.predict(X_test)
	writeResults(X_test, old_label(Yhat_test))

	acc = accuracy(Y_train, Yhat)
	print("Train Accuracy %.4f" % acc)

if __name__ == '__main__':
	main()
