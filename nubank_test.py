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

# dealing with dates
train['last_payment_y'] = pd.to_datetime(train['last_payment']).dt.year.astype(basestring)
train['last_payment_m'] = pd.to_datetime(train['last_payment']).dt.month.astype(basestring)
train['last_payment_d'] = pd.to_datetime(train['last_payment']).dt.day.astype(basestring)
train['end_last_loan_y'] = pd.to_datetime(train['end_last_loan']).dt.year.astype(basestring)
train['end_last_loan_m'] = pd.to_datetime(train['end_last_loan']).dt.month.astype(basestring)
train['end_last_loan_d'] = pd.to_datetime(train['end_last_loan']).dt.day.astype(basestring)
train = train.drop(['last_payment', 'end_last_loan'], axis=1)

# drop columns with too many NAs or too many factors (categorical variables), rows with NA
train = train.drop(['ok_since',
                    'credit_limit',
                    'sign',
                    'n_issues',
                    'job_name',
                    'state',
                    'zip',
                    'reason',
                    'channel'], axis=1)
train = train.dropna()

# get list of ids
ids = list(train['ids'])

# get default values
Y = list(train['default'].values)
train = train.drop(['ids', 'default'], axis=1)

num_train = train.ix[:, train.dtypes == float] # Get numerical features
cat_train = train.ix[:, train.dtypes == object] # Get categorical features

# encode categorical data
le_data = np.empty(cat_train.shape)
list_le = []
for col in range(cat_train.shape[1]):
	le = LabelEncoder()
	le_data[:, col] = le.fit_transform(cat_train.ix[:, col])
	list_le.append(le)

enc = OneHotEncoder(handle_unknown='ignore')
cat_train = enc.fit_transform(le_data).toarray()
# cat_train = pd.DataFrame(cat_train)

# inpute missing numerical data
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
num_train = imp.fit_transform(num_train)
# num_train = pd.DataFrame(num_train)

# put everything back together
X = pd.concat([pd.DataFrame(num_train), pd.DataFrame(cat_train)], axis=1)

# random forest classifier
estimators = 50
classifier = RandomForestClassifier(n_estimators=estimators)
classifier.fit(X, Y)
Yhat = classifier.predict_proba(X)
Yhat = [y[1] for y in Yhat]

# make predictions
test = pd.read_csv(test_file, header=0)
ids_test = list(test['ids'])
test = test.drop(['ids'], axis=1)

test['last_payment_y'] = pd.to_datetime(test['last_payment']).dt.year.astype(basestring)
test['last_payment_m'] = pd.to_datetime(test['last_payment']).dt.month.astype(basestring)
test['last_payment_d'] = pd.to_datetime(test['last_payment']).dt.day.astype(basestring)
test['end_last_loan_y'] = pd.to_datetime(test['end_last_loan']).dt.year.astype(basestring)
test['end_last_loan_m'] = pd.to_datetime(test['end_last_loan']).dt.month.astype(basestring)
test['end_last_loan_d'] = pd.to_datetime(test['end_last_loan']).dt.day.astype(basestring)
test = test.drop(['last_payment', 'end_last_loan'], axis=1)

# drop columns with too many NAs or too many factors (categorical variables), rows with NA
test = test.drop(['ok_since',
                  'credit_limit',
                  'sign',
                  'n_issues',
                  'job_name',
                  'state',
                  'zip',
                  'reason',
                  'channel'], axis=1)

num_test = test.ix[:, test.dtypes == float]  # Get numerical features
cat_test = test.ix[:, test.dtypes == object]  # Get categorical features

le_data = np.empty(cat_test.shape)
for col in range(cat_test.shape[1]):
	le = list_le[col]
	le_data[:, col] = le.fit_transform(cat_test.ix[:, col])

cat_test = enc.transform(le_data).todense()

num_test = imp.transform(num_test)

X_test = pd.concat([pd.DataFrame(num_test), pd.DataFrame(cat_test)], axis=1)

Yhat_test = classifier.predict_proba(X_test)
Yhat_test = [y[1] for y in Yhat_test]

# write to file
with open("predictions.csv", "w") as outfile:
	for i in range(len(Yhat_test)):
		outfile.write("{},{}".format(ids_test[i], Yhat_test[i])+os.linesep)

sum([y>0.5 for y in Yhat])