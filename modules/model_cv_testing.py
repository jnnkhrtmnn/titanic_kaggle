# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:56:22 2019

@author: JannikHartmann
"""

    
def cv_testing(X, y, model, cv=5):

    import numpy as np
    from sklearn.model_selection import cross_val_score


    acc = (cross_val_score(model, X, y, cv=cv, scoring='accuracy'))
    brier = (cross_val_score(model, X, y, cv=cv, scoring='brier_score_loss'))
    
    print(" Mean of Accs: %s" % (np.mean(acc)))
    print(" Std of Accs: %s" % (np.std(acc)))
    
    
    print(" Mean of Brier: %s" % (np.mean(brier)))
    print(" Std of Brier: %s" % (np.std(brier)))
    
    return acc, brier