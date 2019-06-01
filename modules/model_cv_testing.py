# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:56:22 2019

@author: JannikHartmann
"""

def model_cv_testing(X, y, ridge_alpha=0.1, cv=5):

    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    
    model = Ridge(alpha=ridge_alpha)
    
    #from xgboost import XGBRegressor
    #model = XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
     #                      colsample_bytree=1, max_depth=10)

    rmses = np.sqrt(-(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')))


    print(" Mean of RMSEs: %s" % (np.mean(rmses)))
    print(" Std of RMSEs: %s" % (np.std(rmses)))
    
    return rmses


#model_cv_testing(X=X, y=y, ridge_alpha=3, cv=5)
    
def cv_testing(X, y, model, cv=5):

    import numpy as np
    from sklearn.model_selection import cross_val_score
    #from sklearn.linear_model import Ridge
    
    #model = Ridge(alpha=ridge_alpha)
    
    #from xgboost import XGBRegressor
    #model = XGBRegressor(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.75,
     #                      colsample_bytree=1, max_depth=10)

    rmses = np.sqrt(-(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')))


    print(" Mean of RMSEs: %s" % (np.mean(rmses)))
    print(" Std of RMSEs: %s" % (np.std(rmses)))
    
    return rmses