# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:25:13 2020

@author: agleo
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

path = r"C:\Users\agleo\Dropbox\Anton\mock data (McG).xlsx"
df = pd.read_excel(path, index_col = 0)
df['SubjID'] = 777
data_all_subjects = df
initParams = [0.5,0.5]


class LinearDiscountMLE:
    
    def __init__(self, initial_params):
        self.initial_params = initial_params
        
    def deldisc(self,initial_params, df):
        r1 = df['o1_r'].values
        e1 = df['o1_e'].values
        r2 = df['o2_r'].values
        e2 = df['o2_e'].values
        choice = df['choice'].values

        chose_1 = choice == 1

        k = initial_params[0]
        beta = initial_params[1]   
        svi = r2 - k*e2
        svb = r1 - k*e1
        esvi = np.exp(beta*svi)
        esvb = np.exp(beta*svb)
        softmax = esvi / (esvi + esvb)
        softmax[chose_1] =  1 - softmax[chose_1]
        
        LL = -np.sum(np.log(softmax))

        return LL
        
    def optifun(self, df):
        bnds = ((0, 1), (None, None))
        results = minimize(self.deldisc, x0 = self.initial_params, bounds =bnds, args=(df,))
        return results
    
    def estimate_parameters(self, df):
       finres = df.groupby('SubjID').apply(self.optifun)
       return finres


linear_discount_mle = LinearDiscountMLE(initial_params=initParams)

result = linear_discount_mle.estimate_parameters(data_all_subjects)

df_res = pd.DataFrame(result.values.tolist(), index=result.index)

df_res = df_res['x'] 
df_res2 = pd.DataFrame(df_res.values.tolist(), index=df_res.index).rename(columns = {0:'k', 1:'beta'})

#print('k=',str(result.values[0].x[0]))
#print('beta=',str(result.values[0].x[1]))
