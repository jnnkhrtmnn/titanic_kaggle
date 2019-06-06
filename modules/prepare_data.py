#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 03:52:16 2019

@author: jannik
"""

import os
import pandas as pd
import numpy as np

def prepare_data(filename, dpath="data", save_filename="titanic_prepared.csv"):



    titanic_data=pd.read_csv(os.path.join(dpath,filename),sep=",")
    
    ##################################
    titanic_data.index = titanic_data['PassengerId']
    
    titanic_data = pd.get_dummies(titanic_data 
                                  ,columns=['Pclass', 'Sex', 'SibSp', 'Embarked']
                                  ,prefix=['Pclass', 'Sex', 'SibSp', 'Embarked'])


    titanic_data = titanic_data.drop(['Ticket', 'PassengerId', 'Name', 'Cabin'], axis=1)

    
    #legData.dep_dt_scd=pd.to_datetime(legData.dep_dt_scd)
    #legData.arr_dt_scd=pd.to_datetime(legData.arr_dt_scd)
    #legData["rotation"]=legData.ac_registration+legData.flt_date.astype(str)
    #legData["rotseq"]=legData.groupby(['rotation']).cumcount()+1
    #legData=legData.sort_values(["ac_registration","dep_dt_scd"]).reset_index(drop=True)
    #legData["prev_rot"]=[None]+list(legData["rotation"][:-1])
    #legData['prev_arr_dt_scd']=[None]+list(legData["arr_dt_scd"][:-1])
    #legData.loc[(legData.rotation!=legData.prev_rot),"prev_arr_dt_scd"]=None
    #legData["plannedGT"]=(legData.dep_dt_scd-legData.prev_arr_dt_scd).astype("timedelta64[m]")
    #legData.loc[pd.isna(legData.plannedGT),"plannedGT"]=240
    #legData["bookPerc_y"]=legData.booked_class_y/legData.seats_class_y
    #legData.bookPerc_y[legData.bookPerc_y>5]=0
    #legData["shortDist"]=(legData.flt_scd_dist<=400).astype(int)

    ###############################
    
    #legData = pd.get_dummies(legData, columns=["ac_subtype"], prefix=["ac_subtype"])
    #legData = pd.get_dummies(legData, columns=["flt_wave"], prefix=["flt_wave"])
    
    ###############################
    #legData['month'] = legData['flt_date'].apply(lambda x: pd.to_datetime(x).month)
    #legData = pd.get_dummies(legData, columns=["month"], prefix=["month"])
    #legData['quarter'] = legData['flt_date'].apply(lambda x: pd.to_datetime(x).quarter)
    #legData = pd.get_dummies(legData, columns=["quarter"], prefix=["quarter"])
    
    
    
    #############################
    titanic_data.to_csv(dpath+"/"+save_filename, sep=";")
    
    return titanic_data
