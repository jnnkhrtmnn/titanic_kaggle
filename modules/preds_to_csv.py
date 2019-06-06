# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:07:26 2019

@author: JannikHartmann
"""

def preds_to_sub_csv(path, preds_dat, ind):

    import numpy as np
    import pandas as pd
    preds_dat = preds_dat.astype(int).astype(str)
    preds_dat = pd.DataFrame(preds_dat, index=ind)
    preds_dat.columns = ['Survived']
    preds_dat.index.names = ['PassengerId']
    
    preds_dat.to_csv(path+"/submission_survived.csv", sep=",")

    return preds_dat