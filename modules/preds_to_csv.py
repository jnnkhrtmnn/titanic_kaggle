# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:07:26 2019

@author: JannikHartmann
"""

def preds_to_sub_csv(path, preds_dat, ind):

    import numpy as np
    import pandas as pd
    preds_dat = preds_dat.astype(int)
    # check if predictions are lower than zero
    preds_dat = np.maximum(preds_dat, 0) 
    preds_dat = pd.DataFrame(preds_dat, index=ind)
    preds_dat.columns = ['delay']
    preds_dat.index.names = ['flt_leg_i']
    
    #preds_dat[preds_dat['delay']<=7] = 0
    
    preds_dat.to_csv(path+"/submission_delay_preds.csv", sep=";")