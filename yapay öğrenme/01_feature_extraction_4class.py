# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 22:15:12 2019

@author: acseckin
"""


import pandas as pd
import numpy as np
import scipy

def getfeature(data):
    fmean=np.mean(data)
    fstd=np.std(data)
    fmax=np.max(data)
    fmin=np.min(data)
    fkurtosis=scipy.stats.kurtosis(data)
    zero_crosses = np.nonzero(np.diff(data > 0))[0]
    fzero=zero_crosses.size/len(data)
    return fmean,fstd,fmax,fmin,fkurtosis,fzero
def extractFeature(raw_data,ws,hop,dfname):
    fmean=[]
    fstd=[]
    fmax=[]
    fmin=[]
    fkurtosis=[]
    fzero=[]
    flabel=[]
    for i in range(ws,len(raw_data),hop):
       m,s,ma,mi,k,z = getfeature(raw_data.iloc[i-ws+1:i,0])
       fmean.append(m)
       fstd.append(s)
       fmax.append(ma)
       fmin.append(mi)
       fzero.append(z)
       fkurtosis.append(k)
       
       flabel.append(dfname)
    rdf = pd.DataFrame(
    {'mean': fmean,
     'std': fstd,
     'max': fmax,
     'min': fmin,
     'kurtosis': fkurtosis,
     'zerocross':fzero,
     'label':flabel
    })
    return rdf
    

df0 = pd.read_csv('input/0.csv', header = None)
df0_rdf=extractFeature(df0,250,10,"0")

df1 = pd.read_csv('input/1.csv', header = None)
df1_rdf=extractFeature(df1,250,10,"1")

df2 = pd.read_csv('input/2.csv', header = None)
df2_rdf=extractFeature(df2,250,10,"2")

df3 = pd.read_csv('input/3.csv', header = None)
df3_rdf=extractFeature(df3,250,10,"3")

df = pd.concat([df0_rdf, df1_rdf, df2_rdf,df3_rdf])

df.to_csv(r'emg_features.csv', index = None, header=True)

