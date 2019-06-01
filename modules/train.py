# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:41:27 2019

@author:  jannik
"""

from modules.prepare_data import prepare_data

dpath="data/"
filename ="train.csv"


y_X = prepare_data(filename)


#y_X = y_X.dropna()

X = y_X.drop('Survived', axis=1)
y = y_X['Survived']

#from sklearn.preprocessing import MinMaxScaler
#scaler_X = MinMaxScaler()
#scaler_X.fit(X)
#X = scaler_X.transform(X)

###############################


from modules.model_cv_testing import cv_testing

from sklearn.neural_network import MLPRegressor


#cv_testing(X, y, model=MLPRegressor(hidden_layer_sizes = [100,100],
#                             activation = "relu",
#                             alpha = 0.001,
#                             solver = 'lbfgs'), cv=10)

###############################
#from sklearn.neural_network import MLPRegressor
#model=MLPRegressor(hidden_layer_sizes = [50,3],
#                             activation = "tanh",
#                             alpha = 0.001,
#                             solver = 'lbfgs')

from xgboost import XGBRegressor
model1 = XGBRegressor(n_estimators=100, learning_rate=0.05, gamma=0.01, subsample=0.75,
                          colsample_bytree=0.8, max_depth=5)


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
                     max_features=10, max_leaf_nodes=10,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=100,
                     n_jobs=None, oob_score=False, random_state=None,
                     verbose=0, warm_start=False)

from sklearn.linear_model import Ridge
model3 = Ridge(alpha=40)

#from sklearn.linear_model import Lasso
#model3 = Lasso(alpha=20.0, max_iter = 1000)

import numpy as np
y = y 

model1.fit(X,y)
model2.fit(X,y)
model3.fit(X,y)



#from sklearn.externals import joblib
# Save to file in the current working directory
#joblib_file = "joblib_model.pkl"  
#joblib.dump(model, joblib_file)

###############################
###############################
###############################

X_test = prepare_data("leg_data_test.csv", y=False, train=False,
                      save_filename="legData_test_prepared.csv")





###############################
X_test_ind = list(X_test.index)

X_test = X_test.fillna(0)


X_test = scaler_X.transform(X_test)


preds1 = model1.predict(X_test)
preds2 = model2.predict(X_test)
preds3 = model3.predict(X_test)


preds = pd.DataFrame(preds1, index=X_test_ind)
preds['1'] = preds2 
preds['2'] = preds3
preds['delay'] = preds[[0, '1', '2']].mean(axis=1)

preds = preds.drop([0,'1','2'], axis=1)

preds['delay'] = preds['delay']
###############################

from modules.preds_to_csv import preds_to_sub_csv
path = "submissions/"
preds_to_sub_csv(path, preds, ind=X_test_ind)





