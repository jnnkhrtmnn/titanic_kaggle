# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:22:17 2019

@author: JannikHartmann
"""


import os
wd = "C:/Users/JannikHartmann/Desktop/ECB DG-S SAT"

# set directory to code rep
os.chdir(wd)

import pandas as pd

df_name = "Raluca_export-2019-03-29.csv"
df = pd.read_csv(df_name)

# cut df to those we want to search



# Set num_results parameter; how many search results to look at
num_results = 5

for j in range(num_results):
    name = 'summary_'+str(j+1)
    df[name] = [None] * len(df)

from bs4 import BeautifulSoup
import requests, re
from time import sleep

# loop over rows of dataset
for i in range(len(df)):
    
    #address = df["strt"][i]+"+"+df["pstl_cd"][i]+"+"+df["cty"][i]
    searchterm = df["nm_entty"][i] #+"+"+address
    
    # URL for searches; english results only
    url = 'http://google.com/search?q='+ searchterm + '&lr=lang_en'

    # Retrieve html source code; including the info we are looking for
    sleep(0.1)
    res = requests.get(url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')

    # human readable 
    #print(soup.prettify())

    # loop over first 3 search results
    for g in range(num_results): 
        if len(soup.find_all('span',{'class':'st'}))>g:
            summary_text = soup.find_all('span',{'class':'st'})[g].text
            # clean text before storing
            df.iloc[i,g+12] = re.sub("[^A-Za-z ]","",summary_text)
        else:
            # impute no if nothing else available
            df.iloc[i,g+12] = 'no'





# Save file        
df.to_csv("12_04_2161_filled_"+df_name)







    