
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cross_validation import KFold, train_test_split
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso, RandomizedLasso
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

