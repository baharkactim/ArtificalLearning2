# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:12:44 2021

@author: acseckin
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler,MaxAbsScaler,LabelEncoder

##############################################################################
#Reading data from pandas
df = pd.read_csv('data/iris.data')
df.head()

df = pd.read_csv('data/iris.data', header=None)
df.head()

col_name = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df.columns = col_name
df.head()


desc = df.describe() 
df.info()
desc = df['sepal length'].describe() 

#Iris Data from Seaborn
iris = sns.load_dataset('iris')
iris.head()
iris.describe()
iris.info()

print(iris.groupby('species').size())
##############################################################################
#visualisation
sns.set(color_codes=True)

sns.pairplot(iris, hue='species', height=3, aspect=1);
#histogram
iris.hist(edgecolor='black', linewidth=1.2, figsize=(12,8));
plt.show();

#violinplot
plt.figure(figsize=(12,8));
plt.subplot(2,2,1)
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species', y='petal_length', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species', y='petal_width', data=iris);

#boxplot
iris.boxplot(by='species', figsize=(12,8));

#scatter matrix
pd.plotting.scatter_matrix(iris, figsize=(12,10))
plt.show()

# pariplot
sns.pairplot(iris, hue="species", diag_kind="kde");

##############################################################################
#Correlation
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')

##############################################################################
#Preprocessing

sc = StandardScaler()
x_std = sc.fit_transform(x)

min_max_scaler = MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)

max_abs_scaler = MaxAbsScaler()
x_maxabs = max_abs_scaler.fit_transform(x)

## Feature Test Data, non-standardized
x[0:5]
# Feature Test Data, standardized.
x_std[0:5]
x_minmax[0:5]
x_maxabs[0:5]

#label encoder
le = LabelEncoder()
df['class']=le.fit_transform(df['class'])
df['class'].unique()

# 