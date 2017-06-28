import pandas as pd
import numpy as np
import os
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer

# files
train_file = "puzzle_train_dataset.csv"
test_file = "puzzle_test_dataset.csv"

# read in data
train = pd.read_csv(train_file, header=0).dropna(subset = ['default'])
ids = list(train['ids'])
Y = list(train['default'].values)
train = train.drop(['ids', 'default'], axis=1)

num_train = train.ix[:, train.dtypes == float]  # Get numerical features

# inpute missing numerical data
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
num_train = imp.fit_transform(num_train)

# random forest classifier
estimators = 50
classifier = RandomForestClassifier(n_estimators=estimators)
classifier.fit(num_train, Y)
Yhat = classifier.predict_proba(num_train)
Yhat = [y[1] for y in Yhat]

# make predictions
test = pd.read_csv(test_file, header=0)
ids_test = list(test['ids'])
test = test.drop(['ids'], axis=1)
num_test = test.ix[:, test.dtypes == float] # Get numerical features
num_test = imp.transform(num_test)
Yhat_test = classifier.predict_proba(num_test)
Yhat_test = [y[1] for y in Yhat_test]

# write to file
with open("predictions.csv", "w") as outfile:
	for i in range(len(Yhat_test)):
		outfile.write("{},{}".format(ids_test[i], Yhat_test[i])+os.linesep)
