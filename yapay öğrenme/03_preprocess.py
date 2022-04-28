# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:26:12 2021

@author: cagda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

sns.set_style("whitegrid")

## preprocessing methods
"""
Standardization / Mean Removal
Min-Max or Scaling Features to a Range
Normalization
Binarization
"""
from sklearn import preprocessing
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

xmeans=X_train.mean(axis=0)

#Standardization / Mean Removal / Variance Scaling
X_scaled = preprocessing.scale(X_train)
X_scaled.mean(axis=0)# mean of each coulmn=0.0
X_scaled.std(axis=0) # standard deviation of each coulmn=1.0

# scaler applied to train data
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
plt.figure(figsize=(8,6))
plt.hist(X_train);
# scaler applied to test data
X_test = [[-1., 4., 0.]]
scaler.transform(X_test)
plt.figure(figsize=(8,6))
plt.hist(X_test);


# MinMaxScaler
# Scale a data to the [0, 1] range
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

# It is not guarentee to scale the unseen data in the range of  [0,1]
X_test = np.array([[-3., -1.,  0.], [2., 1.5, 4.]])
X_test_minmax = min_max_scaler.transform(X_test)

# MaxAbsScaler
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)

X_test = np.array([[ -1., -0.5,  2.], [0., 0.5, -0.6]])
X_test_maxabs = max_abs_scaler.transform(X_test)

#Normalization
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

normalizer = preprocessing.Normalizer().fit(X) 
normalizer.transform(X)

normalizer.transform([[-1.,  1., 0.]])  

# Binarization
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)
binarizer.transform(X)

# adjusting the binarization threshold
binarizer = preprocessing.Binarizer(threshold=-0.5)

#encoding
#label encoder
source = ['australia', 'singapore', 'new zealand', 'hong kong']
label_enc = preprocessing.LabelEncoder()
src = label_enc.fit_transform(source)

print("country to code mapping:\n") 
for k, v in enumerate(label_enc.classes_): 
    print(v,'\t', k)

test_data = ['hong kong', 'singapore', 'australia', 'new zealand']
result = label_enc.transform(test_data) 
print(result)

#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
src = src.reshape(len(src), 1)
one_hot = one_hot_enc.fit_transform(src)
print(one_hot)
# inverting the code to category label
invert_res = label_enc.inverse_transform([np.argmax(one_hot[0, :])])
print(invert_res)
invert_res = label_enc.inverse_transform([np.argmax(one_hot[3, :])])
print(invert_res)


#preprocessing application

# import dataset
boston_data = load_boston()

df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df.head()

# defining the input and output
X = df[['LSTAT']].values # converts dataframe to numpy array
y = boston_data.target 

# scatter plot input vs output
plt.figure(figsize=(8,6))
plt.scatter(X, y);

# Before Scaling
plt.figure(figsize=(8,6))
plt.hist(X);
plt.xlim(-40, 40);

## without pre-processing
# simple gradient descent regression algorithm
alpha = 0.0001
w_ = np.zeros(1 + X.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X, w_[1:]) + w_[0]
    errors = (y - y_pred)
    
    w_[1:] += alpha * X.T.dot(errors)
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)

# plot the error according to ephoc
plt.figure(figsize=(8,6))
plt.plot(range(1, n_ + 1), cost_);
plt.ylabel('SSE');
plt.xlabel('Epoch');  

## with pre-processing
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1)).flatten()

#After Scaling
plt.figure(figsize=(8,6))
plt.hist(X_std);
plt.xlim(-4, 4);

# simple gradient descent regression algorithm
alpha = 0.0001
w_ = np.zeros(1 + X_std.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X_std, w_[1:]) + w_[0]  # input variable changes as X_std
    errors = (y_std - y_pred)               # input variable changes as y_std
    
    w_[1:] += alpha * X_std.T.dot(errors)   # input variable changes as X_std 
    w_[0] += alpha * errors.sum()
    
    cost = (errors**2).sum() / 2.0
    cost_.append(cost)
plt.figure(figsize=(8,6))
plt.plot(range(1, n_ + 1), cost_);
plt.ylabel('SSE');
plt.xlabel('Epoch');    



