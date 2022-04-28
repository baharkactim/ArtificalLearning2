# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:03:26 2021

Boston Housing Price Prediction

@author: acseckin
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
##############################################################################
# read data

df = pd.read_csv("data/housing.data", delim_whitespace=True, header=None)
df.head()

col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name
df.head()

df.describe()

##############################################################################
# visualize
sns.pairplot(df, height=1.5);
plt.show()

# focus on some features
col_study = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM']
sns.pairplot(df[col_study], height=2.5);
plt.show()

##############################################################################
#Correlation Analysis and Feature Selection
pd.options.display.float_format = '{:,.2f}'.format
df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

#focus on some features
plt.figure(figsize=(12,8))
sns.heatmap(df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'MEDV']].corr(), annot=True, fmt=".2f")
plt.show()

##############################################################################
#Linear Regression with Scikit-Learn
df.head()
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

model = LinearRegression()
model.fit(X, y)
model.coef_
model.intercept_

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();


sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', height=8);
plt.show();

model.predict(np.array([7]).reshape(1,-1))

model_2 = LinearRegression()
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
model_2.fit(X, y)
model_2.predict(np.array([15]).reshape(1,-1))
plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();
sns.jointplot(x='LSTAT', y='MEDV', data=df, kind='reg', height=8);
plt.show();

##############################################################################

"""
Robust Regression http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression

Each iteration performs the following steps:
1- Select min_samples random samples from the original data and check whether 
the set of data is valid (see is_data_valid).
2- Fit a model to the random subset (base_estimator.fit) and check whether the 
estimated model is valid (see is_model_valid).
3- Classify all data as inliers or outliers by calculating the residuals to 
the estimated model (base_estimator.predict(X) - y) - all data samples with 
absolute residuals smaller than the residual_threshold are considered as 
inliers.
4- Save fitted model as best model if number of inlier samples is maximal. In 
case the current estimated model has the same number of inliers, it is only 
considered as the best model if it has better score.
"""
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values
ransac = RANSACRegressor()
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))

sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,8));
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper left')
plt.show()

ransac.estimator_.coef_
ransac.estimator_.intercept_


ransac_2 = RANSACRegressor()
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(0, 40, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))

sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,8));
plt.scatter(X[inlier_mask], y[inlier_mask], 
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper right')
plt.show()

##############################################################################
#Performance Evaluation of Regression Model
X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Residual Analysis
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])
plt.show()

# Mean Squared Error (MSE)
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_test, y_test_pred)

# Coefficient of Determination,  R2
r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)