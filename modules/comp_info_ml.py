# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:06:43 2019

@author: hartmaj
"""


import os
wd = "C:/Users/JannikHartmann/Desktop/ECB DG-S SAT"

# set directory to code rep
os.chdir(wd)

df_name = "15_04_only_CA_S11_S12X_y_features"

import pandas as pd
y_X = pd.read_csv(df_name, index_col=0)

y = y_X["y"]
X = y_X.drop(["y"], axis=1)

ind_x = y_X.index

X.index = pd.Index(range(len(X)))
X_cat = X.select_dtypes(include=[object])

onehotlabels = pd.get_dummies(X_cat, columns=["lgl_frm", "ecnmc_actvty"], prefix=["lgl_frm", "ecnmc_actvty"])

X = X.select_dtypes(include=['float64'])

X = X.join(onehotlabels)
X.index = ind_x.values




# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



###############################################################################
###############################################################################
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV

index = ["RF", "SVM", "NB", "MLP", "XGB"]
results = pd.DataFrame(index = index, columns=["accuracy","precision", "recall", "f1"])

preds_mat = pd.DataFrame(columns=["actual"] + index)
preds_mat["actual"] = y_test

# try random forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=800, max_depth=100, criterion='entropy',
                             oob_score=True)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)

prf = precision_recall_fscore_support(y_test, y_preds, average='macro')
results.loc['RF',:] = [accuracy_score(y_test, y_preds), prf[0], prf[1], prf[2]]
preds_mat["RF"] = y_preds


from sklearn import svm
clf = svm.SVC(C=0.2, class_weight=None, coef0=4.0,
    decision_function_shape='ovr', degree=4, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, verbose=False)

clf.fit(X_train, y_train)  

y_preds = clf.predict(X_test)
prf = precision_recall_fscore_support(y_test, y_preds, average='macro')
results.loc['SVM',:] = [accuracy_score(y_test, y_preds), prf[0], prf[1], prf[2]]
preds_mat["SVM"] = y_preds


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)
prf = precision_recall_fscore_support(y_test, y_preds, average='macro')
results.loc['NB',:] = [accuracy_score(y_test, y_preds), prf[0], prf[1], prf[2]]
preds_mat["NB"] = y_preds


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', activation='tanh', alpha=1e-5, learning_rate='adaptive',
                    hidden_layer_sizes=(800, 5), random_state=1, max_iter=800)

clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)
prf = precision_recall_fscore_support(y_test, y_preds, average='macro')
results.loc['MLP',:] = [accuracy_score(y_test, y_preds), prf[0], prf[1], prf[2]]
preds_mat["MLP"] = y_preds



from xgboost import XGBClassifier

 
clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.9, gamma=0.0, learning_rate=0.1, max_delta_step=0,
       max_depth=50, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=5, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)
prf = precision_recall_fscore_support(y_test, y_preds, average='macro')
results.loc['XGB',:] = [accuracy_score(y_test, y_preds), prf[0], prf[1], prf[2]]
preds_mat["XGB"] = y_preds

# combination by majority vote
preds_mat_int = preds_mat[index].copy()
preds_mat_int.replace(to_replace=dict(s12x=1, s11=0), inplace=True)
preds_mat_int = preds_mat_int.apply(pd.to_numeric, errors = 'coerce')
preds_mat_int['majority_vote'] = preds_mat_int.sum(axis=1)
preds_mat_int.loc[preds_mat_int['majority_vote'] < 3, 'majority_vote'] = 0
preds_mat_int.loc[preds_mat_int['majority_vote'] > 2, 'majority_vote'] = 1
preds_mat_int['majority_vote'].replace([0,1],['s11','s12x'],inplace=True)
preds_mat['majority_vote'] = preds_mat_int['majority_vote']

y_preds = preds_mat['majority_vote']
prf = precision_recall_fscore_support(y_test, y_preds, average='macro')
results.loc['Vote',:] = [accuracy_score(y_test, y_preds), prf[0], prf[1], prf[2]]

print(results)


################# Analysis of preds_mat; when preds are wrong, are all predictions the same?

preds_mat[(preds_mat['actual']=='s12x') & (preds_mat['majority_vote']=='s11') & (preds_mat['SVM']=='s11') & (preds_mat['MLP']=='s11')]
preds_mat[(preds_mat['actual']=='s11') & (preds_mat['majority_vote']=='s12x') & (preds_mat['SVM']=='s12x') & (preds_mat['MLP']=='s12x')]


'''
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.neural_network import MLPClassifier


parameters = {'solver': ['lbfgs'], 
              'max_iter': [200, 600, 1200], 
              'alpha': 10.0 ** -np.arange(1, 6), 
              'hidden_layer_sizes':[(100, 2), (100, 10),(800, 5),(1600, 5),(2000, 2)], 
              'random_state':[1],
              'learning_rate':['adaptive','constant'],
              'activation': ['identity', 'logistic', 'tanh', 'relu'],  
              }
#, 'invscaling'
clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=1)
clf.fit(X_train, y_train)
'''