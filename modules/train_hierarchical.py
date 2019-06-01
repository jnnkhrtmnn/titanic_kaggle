# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:30:47 2019

@author: JannikHartmann
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:41:27 2019

@author: U724080
"""

from modules.prepare_data import prepare_data

dpath="data_sets/"
filename ="leg_data_train.csv"


y_X = prepare_data(filename,train=True)
'''
import pandas as pd
y_X = pd.read_csv(dpath+'legData_train_prepared.csv', sep=";")


column_select = [
       'seats_class_y',
       'pax_children', 'pax_infant',
       'bookPerc_y',
       'delay', 'rotseq',
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
       'windspeed', 'snow', 'light_snow', 'snow_grains',
       'thunderstorm']


y_X = y_X[column_select]
'''
y_X = y_X[y_X['quarter_2']==0]
y_X = y_X[y_X['quarter_3']==0]


y_X = y_X.dropna()

X = y_X.drop('delay', axis=1)
y = y_X['delay']

from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_X.fit(X)
X = scaler_X.transform(X)


###############################


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, max_depth=20, criterion='entropy',
                             oob_score=False)

y_class = y.copy()
y_class.loc[y < 4] = 0
y_class.loc[y >= 4] = 1

clf.fit(X, y_class)


from sklearn.linear_model import Ridge
model1 = Ridge(alpha=0.001)
model2 = Ridge(alpha=0.001)

#from sklearn.neural_network import MLPRegressor
#model1=MLPRegressor(hidden_layer_sizes = [100,2],
#                             activation = "relu",
#                             alpha = 0.05,
#                             solver = 'lbfgs')

#model2=MLPRegressor(hidden_layer_sizes = [100,2],
#                             activation = "relu",
#                             alpha = 0.05,
#                             solver = 'lbfgs')

import pandas as pd
X = pd.DataFrame(X, index=y.index)

use1 = X.index[y_class==0].tolist()
use2 = X.index[y_class==1].tolist()

model1.fit(X.loc[use1],y[use1])
model2.fit(X.loc[use2],y[use2])



###############################
###############################
###############################

X_test = prepare_data("leg_data_test.csv", y=False, train=False,
                      save_filename="legData_test_prepared.csv")

'''
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
       'windspeed', 'snow', 'light_snow', 'snow_grains',
       'thunderstorm']

X_test = X_test[column_select_test]
'''
###############################
X_test_ind = list(X_test.index)

X_test = X_test.fillna(0)


X_test = scaler_X.transform(X_test)


use = clf.predict(X_test)

import pandas as pd

X_test = pd.DataFrame(X_test, index= X_test_ind)

use1 = X_test.index[use==0].tolist()
use2 = X_test.index[use==1].tolist()

preds1 = model1.predict(X_test.loc[use1])
preds2 = model2.predict(X_test.loc[use2])

preds1 = pd.DataFrame(preds1, index=X_test.loc[use1].index)
preds2 = pd.DataFrame(preds2, index=X_test.loc[use2].index)

preds = preds1.append(preds2)

###############################

from modules.preds_to_csv import preds_to_sub_csv
path = "submissions/"
preds_to_sub_csv(path, preds, ind=X_test_ind)






