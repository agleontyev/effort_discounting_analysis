# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:06:12 2019

@author: agleo
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

path = r"C:\mock data (McG).xlsx"
df = pd.read_excel(path, index_col = 0)
df['SubjID'] = 777
data_all_subjects = df

def optifun(df):
    r1 = df['o1_r'].values
    e1 = df['o1_e'].values
    r2 = df['o2_r'].values
    e2 = df['o2_e'].values
    choice = df['choice'].values

    chose_1 = choice == 1

    def deldisc(params):
        k = params[0]
        beta = params[1]   
        svi = r2 - k*e2
        svb = r1 - k*e1
        esvi = np.exp(beta*svi)
        esvb = np.exp(beta*svb)
        softmax = esvi / (esvi + esvb)
        softmax[chose_1] =  1 - softmax[chose_1]
        
        LL = -np.sum(np.log(softmax))

        return(LL)

    initParams = [0.5,0.5]
    results = minimize(deldisc, initParams)
    return results

results = data_all_subjects.groupby('SubjID').apply(optifun)

print('k=',str(results.values[0].x[0]))
print('beta=',str(results.values[0].x[1]))
