# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:41:27 2019

@author:  jannik
"""
###############################################################################

from modules.prepare_data import prepare_data
import pandas as pd
import os

dpath="data/"
filename_train ="train.csv"
filename_test ="test.csv"
    
y_X = pd.read_csv(os.path.join(dpath,filename_train),sep=",")
    
y_X = y_X.append(pd.read_csv(os.path.join(dpath,filename_test),sep=","), sort=True)

y_X.index = y_X['PassengerId']

X = y_X.drop('Survived', axis=1)
y = y_X['Survived']

# drop unused vars
X = X.drop(['Ticket', 'PassengerId', 'Name', 'Cabin'], axis=1)

# get dummies for categorical variables    
X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'SibSp', 'Embarked']
                    , prefix=['Pclass', 'Sex', 'SibSp', 'Embarked'])

# impute age with median if missing
X['Age'] = X['Age'].fillna(X['Age'].median())


#X = X.dropna()

# Names








#from sklearn.preprocessing import MinMaxScaler
#scaler_X = MinMaxScaler()
#scaler_X.fit(X)
#X = scaler_X.transform(X)


###############################################################################

# Use cross validation to choose features. models and parameters
from modules.model_cv_testing import cv_testing

from sklearn.neural_network import MLPClassifier
cv_testing(X, y, model=MLPClassifier(hidden_layer_sizes = [30,10],
                             activation = "logistic",
                             alpha = 0.005,
                             solver = 'lbfgs'), cv=5)


from sklearn.tree import DecisionTreeClassifier
cv_testing(X, y, model=DecisionTreeClassifier(criterion='gini'
                                              ,max_depth=3
                                              ,min_samples_leaf=1
                                              ,min_samples_split=0.1
                                              ,random_state=None
                                              ,max_features=None
                                              ), cv=5)


###############################################################################
###############################################################################

# create table for storing predictions
index = ["RF", "SVM", "NB", "MLP", "XGB"]
results = pd.DataFrame(index = index, columns=["accuracy","precision", "recall", "f1"])
preds_mat = pd.DataFrame(columns=index)


###############################################################################

# define models
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=100, max_depth=100, criterion='entropy',
                             oob_score=True)

from sklearn import svm
model2 = svm.SVC(C=0.2, class_weight=None, coef0=4.0,
    decision_function_shape='ovr', degree=4, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, verbose=False)

from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()

from sklearn.neural_network import MLPClassifier
model4 = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.005,
                    hidden_layer_sizes=(30, 10))

from xgboost import XGBClassifier 
model5 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.9, gamma=0.0, learning_rate=0.1, max_delta_step=0,
       max_depth=50, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=5, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

models = [model1, model2, model3, model4, model5]

###############################################################################
  
# preprocess test data
X_test = prepare_data("test.csv", save_filename="test_prepared.csv")
X_test_ind = list(X_test.index)


X_test = X_test.fillna(0)


###############################################################################

# fit models and store predictions
for i in range(len(models)):
    model = models[i]
    model.fit(X,y)
    preds_mat[index[i]] = model.predict(X_test)

    
###############################################################################

# combination of predictions
preds_mat = preds_mat.apply(pd.to_numeric, errors = 'coerce')
preds_mat['majority_vote'] = preds_mat.mean(axis=1)
preds_mat.loc[preds_mat['majority_vote'] < 0.5, 'majority_vote'] = 0
preds_mat.loc[preds_mat['majority_vote'] >= 0.5, 'majority_vote'] = 1
preds_mat.index = X_test_ind

###############################################################################

# store submission in correct format
from modules.preds_to_csv import preds_to_sub_csv
path = "submissions/"
preds_to_sub_csv(path, preds_mat['majority_vote'], ind=X_test_ind)





