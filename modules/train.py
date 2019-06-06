# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:41:27 2019

@author:  jannik
"""
###############################################################################

import pandas as pd
import os

dpath="data/"
filename_train ="train.csv"
filename_test ="test.csv"
    
y_X = pd.read_csv(os.path.join(dpath,filename_train),sep=",")
    
len_train = len(y_X)

y_X = y_X.append(pd.read_csv(os.path.join(dpath,filename_test),sep=","), sort=True)

y_X.index = y_X['PassengerId']

X = y_X.drop('Survived', axis=1)
y = y_X['Survived']


###############################################################################

X.describe()

# drop unused vars
X = X.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1)


# impute age with median if missing
X['Age'] = X['Age'].fillna(X['Age'].median())

#import seaborn as sns
#sns.distplot(X['Age'])
#set up bins
bins = [-1,16,50,100]
age_labels = ['young', 'mid', 'old']
categ = pd.cut(X.Age,bins, labels=age_labels)
categ = pd.DataFrame(categ)
categ.columns = ['Age_categ']
X = X.join(categ)
X = X.drop(['Age'], axis=1)




titles_raw = set()
for x_name in X['Name']:
    titles_raw.add(x_name.split(',')[1].split('.')[0].strip())

X['Title'] = X['Name'].map(lambda x_name : x_name.split(',')[1].split('.')[0].strip())

# use title to define groups
title_grouping = {'Jonkheer' : 'nobile'
                  ,'the Countess' : 'nobile'
                  ,'Sir' : 'nobile'
                  ,'Lady' : 'nobile'
                  ,'Don' : 'nobile'
                  ,'Dona' : 'nobile'
                  ,'Capt' : 'military'
                  ,'Col' : 'military'
                  ,'Major' : 'military'
                  ,'Rev' : 'military'
                  ,'Mr' : 'Mr'
                  ,'Mrs' : 'Mrs'
                  ,'Ms' : 'Ms'
                  ,'Mme' : 'Mrs'
                  ,'Mlle' : 'Ms'
                  ,'Miss' : 'Ms'
                  ,'Dr' : 'other'
                  ,'Master' : 'other'}
    
X['Title'] = X.Title.map(title_grouping)

X = X.drop(['Name'], axis=1)

X['Fare'].describe()
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# fares
#import seaborn as sns
#sns.distplot(X['Fare'])
# maybe built some buckets; has not been successful


# sibsp
#set up bins
bins = [-1,0,4,9]
SibSp_labels = ['0', 'few', 'large']
categ = pd.cut(X.SibSp,bins, labels=SibSp_labels)
categ = pd.DataFrame(categ)
categ.columns = ['SibSp_categ']
#concatenate age and its bin
X = X.join(categ)
X = X.drop(['SibSp'], axis=1)


# get dummies for categorical variables    
X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'SibSp_categ', 'Embarked', 'Title', 'Age_categ']
                    , prefix=['Pclass', 'Sex', 'SibSp_categ', 'Embarked', 'Title', 'Age_categ'])


X = X.dropna()


#from sklearn.preprocessing import MinMaxScaler
#scaler_X = MinMaxScaler()
#scaler_X.fit(X)
#X = pd.DataFrame(scaler_X.transform(X))


###############################################################################

X_train, X_test, y_train = X.iloc[:len_train,], X.iloc[len_train:,], y.iloc[:len_train,]

# Use cross validation to choose features. models and parameters
from modules.model_cv_testing import cv_testing

from sklearn.ensemble import RandomForestClassifier
cv_testing(X_train, y_train, model=RandomForestClassifier(n_estimators=200, 
                                                          min_samples_split=4, 
                                                          criterion='entropy',
                                                          oob_score=False), cv=5)

from sklearn import svm
cv_testing(X_train, y_train, model=svm.SVC(C=0.2, class_weight=None, coef0=8.0, 
                                           degree=2, gamma=0.01, 
                                           kernel='poly', 
                                           probability=True), cv=5)


from sklearn.neural_network import MLPClassifier
cv_testing(X_train, y_train, model=MLPClassifier(hidden_layer_sizes = [20,5],
                             activation = "logistic",
                             alpha = 0.01,
                             solver = 'lbfgs'), cv=5)

from xgboost import XGBClassifier 
cv_testing(X_train, y_train, model=XGBClassifier(base_score=0.5, booster='gbtree', 
                                                 colsample_bylevel=1,
                                                 colsample_bytree=0.9, 
                                                 gamma=0.0, learning_rate=0.1, 
                                                 max_delta_step=0,
                                                 max_depth=50, 
                                                 min_child_weight=1,
                                                 n_estimators=30,
                                                 objective='binary:logistic', 
                                                 reg_alpha=0, reg_lambda=5, 
                                                 scale_pos_weight=1, 
                                                 subsample=1), cv=5)



###############################################################################
###############################################################################

# create table for storing predictions
index = ["RF", "SVM", "MLP", "XGB"]
preds_mat = pd.DataFrame(columns=index)


###############################################################################

# define models
from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=500, max_depth=10, criterion='entropy',
                             oob_score=True)

from sklearn import svm
model2 = svm.SVC(C=0.2, class_weight=None, coef0=4.0,
    decision_function_shape='ovr', degree=2, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, verbose=False)

from sklearn.neural_network import MLPClassifier
model3 = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.005,
                    hidden_layer_sizes=(30, 10))

from xgboost import XGBClassifier 
model4 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.9, gamma=0.0, learning_rate=0.1, max_delta_step=0,
       max_depth=50, min_child_weight=1, missing=None, n_estimators=800,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=5, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

models = [model1, model2, model3, model4]

###############################################################################
  
# get test data index
X_test_ind = list(X_test.index)


X_test = X_test.fillna(0)


###############################################################################

# fit models and store predictions
for i in range(len(models)):
    model = models[i]
    model.fit(X_train, y_train)
    preds_mat[index[i]] = model.predict(X_test)

    
###############################################################################

# combination of predictions
preds_mat.index = X_test_ind
preds_mat['majority_vote'] = preds_mat.mean(axis=1)
preds_mat.loc[preds_mat['majority_vote'] < 0.5, 'majority_vote'] = 0
preds_mat.loc[preds_mat['majority_vote'] >= 0.5, 'majority_vote'] = 1


###############################################################################

# store submission in correct format
from modules.preds_to_csv import preds_to_sub_csv
path = "submissions/"
preds_to_sub_csv(path, preds_mat['majority_vote'], ind=X_test_ind)





