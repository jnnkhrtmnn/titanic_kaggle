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
X = X.drop(['PassengerId', 'Ticket'], axis=1)


# impute age with median if missing
X['Age'] = X['Age'].fillna(X['Age'].median())

#import seaborn as sns
#sns.distplot(X['Age'][:len_train][y==1], hist=False)
#sns.distplot(X['Age'][:len_train][y==0], hist=False)


#set up bins
bins = [-1,18,35,48,100]
age_labels = ['young', 'mid_young', 'mid_old', 'old']
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

y.groupby(X['Title']).mean()



X['Fare'].describe()
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# fares
#import seaborn as sns
#sns.distplot(X['Fare'][:len_train][y==1], hist=False)
#sns.distplot(X['Fare'][:len_train][y==0], hist=False)

# maybe built some buckets; has not been successful
bins = [-1, 5, 10, 50, 70, 1000]
fares_labels = [0,1,2,3,4]
categ = pd.cut(X.Fare, bins, labels=fares_labels)
categ = pd.DataFrame(categ)
categ.columns = ['Fare_categ']

#y.groupby(X['Fare_categ']).mean()
X = X.join(categ)
X['Fare_categ'] = X['Fare_categ'].astype(int)
X = X.drop(['Fare'], axis=1)


# sibsp join with Parch
X['family'] = X['Parch'] + X['SibSp']

y.groupby(X['family']).mean()

#set up bins
bins = [-1,0,3,10]
fam_labels = ['0', 'few', 'large']
categ = pd.cut(X.family, bins, labels=fam_labels)
categ = pd.DataFrame(categ)
categ.columns = ['fam_categ']
#concatenate age and its bin
X = X.join(categ)
X = X.drop(['SibSp', 'Parch', 'family'], axis=1)



X['Cabin'] = X['Cabin'].fillna('U')

cabins = set()
for row in X['Cabin']:
    cabins.add(row[0].strip())

X['Cabin'] = X['Cabin'].map(lambda row : row[0].strip())

y.groupby(X['Cabin']).mean()
# merge B, D, E and C,F
X.loc[X['Cabin'].isin(['B', 'D', 'E']), 'Cabin'] = 'BDE'
X.loc[X['Cabin'].isin(['C', 'F']), 'Cabin'] = 'CF'



# get dummies for categorical variables    
X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'fam_categ', 'Embarked', 'Title', 'Age_categ', 'Cabin', 'Fare_categ']
                    , prefix=['Pclass', 'Sex', 'fam_categ', 'Embarked', 'Title', 'Age_categ', 'Cabin', 'Fare_categ'])


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
rf_params = {'n_estimators' : 500
            ,'min_samples_split' : 7
            ,'criterion' : 'gini'}

cv_testing(X_train, y_train, model=RandomForestClassifier(**rf_params), cv=5)

from sklearn import svm
svm_params = {'C' : 12
             ,'class_weight' : None
             ,'coef0' : 1.0
             ,'degree' : 2
             ,'gamma' : 0.05
             ,'kernel' : 'poly'
             ,'probability' : True}

cv_testing(X_train, y_train, model=svm.SVC(**svm_params), cv=5)


from sklearn.neural_network import MLPClassifier
mlp_params = {'hidden_layer_sizes' : [20,5]
             ,'activation' : 'logistic'
             ,'alpha' : 0.01
             ,'solver' : 'lbfgs'}

cv_testing(X_train, y_train, model=MLPClassifier(**mlp_params), cv=5)

from xgboost import XGBClassifier 
xgb_params = {'base_score' : 0.5
             ,'booster' : 'gbtree' 
             ,'colsample_bylevel' : 1
             ,'colsample_bytree' : 0.9
             ,'gamma' : 0.0
             ,'learning_rate' : 0.1
             ,'max_delta_step' : 0
             ,'max_depth' : 50 
             ,'min_child_weight' : 1
             ,'n_estimators' : 50
             ,'objective' : 'binary:logistic' 
             ,'reg_alpha' : 0.2
             ,'reg_lambda' : 2 
             ,'scale_pos_weight' : 1
             ,'subsample' : 1}

cv_testing(X_train, y_train, model=XGBClassifier(**xgb_params), cv=5)


from sklearn.linear_model import LogisticRegression
lr_params = {'penalty' : 'l2'
            ,'C' : 1
            ,'solver' : 'liblinear'}

cv_testing(X_train, y_train, model=LogisticRegression(**lr_params), cv=5)


from sklearn.neighbors import KNeighborsClassifier
knn_params = {'n_neighbors' : 3}

cv_testing(X_train, y_train, model=KNeighborsClassifier(**knn_params), cv=5)

###############################################################################
###############################################################################

# create table for storing predictions
index = ['RF', 'MLP', 'XGB', 'SVM', 'LR']
preds_mat = pd.DataFrame(columns=index)


###############################################################################

# define models
model1 = RandomForestClassifier(**rf_params)
model2 = MLPClassifier(**mlp_params) 
model3 = XGBClassifier(**xgb_params)
model4 = svm.SVC(**svm_params)
model5 = LogisticRegression(**lr_params)

models = [model1, model2, model3]

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





