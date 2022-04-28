# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 22:02:36 2021

@author: cagda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
df.head()

df.shape

# assign the inputs and outputs 
X = df
y = boston_data.target

# Stats models
import statsmodels.api as sm
import statsmodels.formula.api as smf

X_constant = sm.add_constant(X)
pd.DataFrame(X_constant)

model = sm.OLS(y, X_constant)
lr = model.fit()
lr.summary()
"""
Model Statistical Outputs

Dep. Variable: The dependent variable or target variable
Model: Highlight the model used to obtain this output. It is OLS here. Ordinary least squares / Linear regression
Method: The method used to fit the data to the model. Least squares
No. Observations: The number of observations
DF Residuals: The degrees of freedom of the residuals. Calculated by taking the number of observations less the number of parameters
DF Model: The number of estimated parameters in the model. In this case 13. The constant term is not included.

Residual Tests
Omnibus D'Angostino's test: This is a combined statistical test for skewness and kurtosis.
Prob(Omnibus): p-value of Omnibus test.
Skewness: This is a measure of the symmetry of the residuals around the mean. Zero if symmetrical. A positive value indicates a long tail to the right; a negative value a long tail to the left.
Kurtosis: This is a measure of the shape of the distribution of the residuals. A normal distribution has a zero measure. A negative value points to a flatter than normal distribution; a positive one has a higher peak than normal distribution.
Durbin-Watson: This is a test for the presence of correlation among the residuals. This is especially important for time series modelling
Jarque-Bera: This is a combined statistical test of skewness and kurtosis.
Prob (JB): p-value of Jarque-Bera.
Cond. No: This is a test for multicollinearity. > 30 indicates unstable results
"""

# Correlation Matrix
pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = df.corr()
# elemination/detection of low correlated features
corr_matrix[np.abs(corr_matrix) < 0.6] = 0

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()

# Detecting Collinearity with Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(df.corr())
pd.options.display.float_format = '{:,.4f}'.format
pd.Series(eigenvalues).sort_values()

#detecting the factors that are causing multicollinearity problem.
np.abs(pd.Series(eigenvectors[:,8])).sort_values(ascending=False)
print(df.columns[2], df.columns[8], df.columns[9])

# Revisiting Feature Importance and Extractions
df.head()
plt.hist(df['TAX']);
plt.hist(df['NOX']);

# Standardise Variable to Identify Key Feature(s)
#before the scaler
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
result = pd.DataFrame(list(zip(model.coef_, df.columns)), columns=['coefficient', 'name']).set_index('name')
np.abs(result).sort_values(by='coefficient', ascending=False)

#after the scaler
from sklearn.preprocessing import StandardScaler  
from sklearn.pipeline import make_pipeline  
scaler = StandardScaler()  
standard_coefficient_linear_reg = make_pipeline(scaler, model)
standard_coefficient_linear_reg.fit(X,y)
result = pd.DataFrame(list(zip(standard_coefficient_linear_reg.steps[1][1].coef_, df.columns)), 
                      columns=['coefficient', 'name']).set_index('name')
np.abs(result).sort_values(by='coefficient', ascending=False)

#Use  R2  to Identify Key Features
from sklearn.metrics import r2_score
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
benchmark = linear_reg.fit()
r2_score(y, benchmark.predict(df))

# without LSTAT feature
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B', 
              data=df)
lr_without_LSTAT = linear_reg.fit()
r2_score(y, lr_without_LSTAT.predict(df))

# without AGE feature
linear_reg = smf.ols(formula = 'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT', 
              data=df)
lr_without_AGE = linear_reg.fit()
r2_score(y, lr_without_AGE.predict(df))
