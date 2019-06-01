# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:41:27 2019

@author:  jannik
"""

from modules.prepare_data import prepare_data

dpath="data/"
filename ="train.csv"


y_X = prepare_data(filename)



y_X = y_X.dropna()

X = y_X.drop('delay', axis=1)
y = y_X['delay']

from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_X.fit(X)
X = scaler_X.transform(X)

###############################

#from modules.model_cv_testing import model_cv_testing
#model_cv_testing(X, y, ridge_alpha=0.1, cv=10)

# Mean of RMSEs: 13.771345123044162 (old features)
# Std of RMSEs: 1.649801050141028
#Out[29]: array([11.7268377 , 13.36392297, 12.97220095, 16.68913631, 14.10462768]) 

#Mean of RMSEs: 13.711375621169804 (new features)
# Std of RMSEs: 1.6058369561414712
#Out[38]: array([11.74045426, 13.26983649, 12.92853419, 16.55326449, 14.06478868]) 

# Mean of RMSEs: 13.704269805010252 (mit dep_from_lis)
# Std of RMSEs: 1.603969665076207
#Out[50]: array([11.73469287, 13.2598078 , 12.92327864, 16.54129237, 14.06227734])

################################

#from modules.model_cv_testing import cv_testing

#from sklearn.neural_network import MLPRegressor


#cv_testing(X, y, model=MLPRegressor(hidden_layer_sizes = [100,100],
#                             activation = "relu",
#                             alpha = 0.001,
#                             solver = 'lbfgs'), cv=10)
#Mean of RMSEs: 13.721167699563235 (old features)
# Std of RMSEs: 1.6441799524327008
#Out[31]: array([11.77190058, 13.19231177, 12.88992621, 16.65459905, 14.09710089]) 

#Mean of RMSEs: 13.665375633728038 (mit dep_from_lis)
# Std of RMSEs: 1.6218607231595141
#Out[53]: array([11.70015943, 13.17001048, 12.84008545, 16.52591558, 14.09070723])

#cv_testing(X, y, model=MLPRegressor(hidden_layer_sizes = [10,10],
#                             activation = "relu",
#                             alpha = 0.001,
#                             solver = 'lbfgs'), cv=10)
#Mean of RMSEs: 13.771264437672377 (mit dep_from_lis)
# Std of RMSEs: 1.5663195997800408
#Out[52]: array([11.80044568, 13.08746704, 13.36709489, 16.54034313, 14.06097144])

#from sklearn.ensemble import RandomForestRegressor
#cv_testing(X, y, model=RandomForestRegressor(n_estimators=10), cv=10)

# Mean of RMSEs: 14.719927094058855 (old features)
# Std of RMSEs: 1.497920854869965
#Out[32]: array([12.59496639, 14.8594358 , 13.68621582, 16.95672304, 15.50229442])

#cv_testing(X, y, model=RandomForestRegressor(n_estimators=50), cv=10)
#Mean of RMSEs: 14.01712647711555 (new features)
# Std of RMSEs: 1.5462067692692467
#Out[40]: array([12.04719163, 14.27523188, 12.8527955 , 16.57652269, 14.33389067])

#from xgboost import XGBRegressor
#cv_testing(X, y, model = XGBRegressor(n_estimators=300, learning_rate=0.08, gamma=0.01, subsample=0.75,
#                           colsample_bytree=1, max_depth=10), cv=10)


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


column_select_test = [
       'seats_class_y',
       'pax_children', 'pax_infant',
       'bookPerc_y',
       'rotseq',
       'plannedGT', 'crew_total', 'crew_sick',
       'crew_reserve', 'outflights', 'inflights', 'Dangerous good',
       'Express good', 'Valuable', 'ac_subtype_100',
       'ac_subtype_319', 'ac_subtype_320', 'ac_subtype_321', 'ac_subtype_32A',
       'ac_subtype_32M', 'ac_subtype_333', 'ac_subtype_33W', 'ac_subtype_343',
       'ac_subtype_77W', 'ac_subtype_AR1', 'ac_subtype_CR9', 'ac_subtype_CS1',
       'ac_subtype_CS3', 'ac_subtype_DH4', 'ac_subtype_E90', 'ac_subtype_S20',
       'flt_wave_ZRH W1', 'flt_wave_ZRH W2', 'flt_wave_ZRH W3',
       'flt_wave_ZRH W4', 'flt_wave_ZRH W5', 'flt_wave_ZRH W6',
       'flt_wave_w/o Wave','month_1', 'month_2', 'month_3',
       'month_4', 'month_5', 'month_6','month_7', 'month_8',
       'month_9', 'month_10', 'month_11', 'month_12',
       'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4', 'dep_from_zrh','dep_from_lis',
       'strong_wind', 'snow', 'light_snow', 'snow_grains',
       'thunderstorm','shortDist']

X_test = X_test[column_select_test]

#X_test["off_season"] = X_test["month_2"]+X_test["month_3"]+X_test["month_4"]+X_test["month_11"]

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





