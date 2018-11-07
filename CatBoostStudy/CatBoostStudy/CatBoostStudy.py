import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


# ipywidgets and ipython"

train = pd.read_csv("allPMData.csv")
train.dtypes
#Creating a training set for modeling and validation set to check model performance
X = train.drop(['AVG_PM_2_5','Year','Month','Day','Hour','CBWD'], axis=1)
y = train.AVG_PM_2_5

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.9, random_state=1234)
model=CatBoostRegressor(iterations=200, depth=10, learning_rate=1, loss_function='RMSE')
model.fit(X_train, y_train,eval_set=(X_validation, y_validation))
model.score(X_validation,y_validation)