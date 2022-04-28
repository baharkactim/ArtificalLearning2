# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 23:08:26 2021

@author: cagda
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

plt.close("all")

housing = pd.read_csv("data/housing.csv",delim_whitespace=True)
print("Veri setinin şekli",housing.shape)
print("veri seti değişken tipleri:",housing.dtypes)
print("veri setinin ilk 10 satırı")
print(housing.head(10))
print("veri setinin istatistiki verileri")
description=housing.describe()
print(description)
# veri seti içindeki değişkenlerin dağılımlarının çizdirilmesi
housing.hist(bins=10,figsize=(16,9),grid=False);

# veri seti içindeki değişkenlerin ilişki katsayılarının çizdirilmesi
print("Veri seti içindeki değişkenlerin birbiri ile ilişki katsayısı")
corr=housing.corr(method='pearson')
plt.figure()
sns.heatmap(corr, annot = True)
# ilişki katsayılarına bakıldığında 'RM', 'LSTAT', 'PTRATIO' değişkenleri MEDV için yüksek ilişkili
#düşük ilişki katsayılı olanları çıkaralım
prices = housing['MEDV']
housing = housing.drop(['CHAS','CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'], axis = 1)
features = housing.drop('MEDV', axis = 1)
print ("Geriye kalan veri seti değerleri")
print(housing.head())

# Verileri çıkış değerine karşılık çizdirme
plt.figure(figsize=(20, 5))
for i, col in enumerate(features.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, len(features.columns), i+1)
    x = housing[col]
    y = prices
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prices')

# verilerin box çizimi

housing.plot(kind='box', subplots=True, layout=(1,6), sharex=False, sharey=False)

# verinin hazırlanması 

rescaledX = StandardScaler().fit_transform(features)
X = pd.DataFrame(data = rescaledX, columns= features.columns)
Y = prices 
validation_size = 0.20
seed = 7
num_folds = 10
RMS = 'neg_mean_squared_error'

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#denenecek modellerin yüklenmesi
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVR-Linear', SVR(kernel="linear")))
models.append(('SVR-RBF', SVR()))
models.append(('SVR-Poly2', SVR(kernel="poly",degree=2)))
models.append(('SVR-Poly3', SVR(kernel="poly",degree=3)))
models.append(('SVR-Sigmoid', SVR(kernel="sigmoid")))
models.append(('ANN-lbfgs',MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))
models.append(('ANN-sgd',MLPRegressor(solver='sgd', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))
models.append(('ANN-adam',MLPRegressor(solver='adam', alpha=1e-5,hidden_layer_sizes=(20,10,5), random_state=7)))

# modellerin sınanması
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=RMS)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
puan=[]
for i in range(len(names)):
    print(names[i],results[i].mean())
    puan.append(results[i].mean())
print("En yüksek doğruluk(Accuracy) değeri:")
print(names[puan.index(max(puan))], max(puan))

# algoritmaların sonuçlarının karşılaştırılmasının çizimi
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.grid()
ax.set_xticklabels(names)
plt.show()

# test verisinin çıkarımı
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

# transform the validation dataset
X_test_rescaled = scaler.transform(X_test)

fig = plt.figure()
fig.suptitle("Algorithm Comparision")

nax=len(models)
i=1
for name, model in models:
    # test verisine karşılık tahmin
    model.fit(rescaledX, Y_train)
    test_predictions = model.predict(X_test_rescaled)
    
    # test verisinin bulunduğu bölgede eğri çizmek için periyodik veri doldurma
    curveaxis=np.zeros((100,X_test_rescaled.shape[1]))
    for cx in range(X_test_rescaled.shape[1]):
        curveaxis[:,cx]=np.linspace(np.min(X_test_rescaled[:,cx]),np.max(X_test_rescaled[:,cx]),100) # linspace komutu başlangıç ve bitiş değerleri arasında belirtilen sayı kadar(100) parçalı değer oluşturur 
    curve_predictions = model.predict(curveaxis) 
    
    #tahmin ve rezidü çizimleri
    print(name,":", mean_squared_error(Y_test, test_predictions))
    plt.subplot(5,3,i) # 5 satır 4 sütun çizim alanında i. çizim
    plt.title(name) # çizim başlığı
    plt.scatter(X_test_rescaled[:,0], Y_test,c='b') # test verisi 
    plt.scatter(X_test_rescaled[:,0], test_predictions,c='r',alpha=0.5) # test verisine karşılık prediction
    plt.plot(curveaxis[:,0], curve_predictions,c='r')# 0 sütunu değer atamaya karşılık tahminlerin eğri olarak çizilmesi
    plt.grid()
    
    i=i+1 # subplot indeksi
    """
    # Bu kısım modelin daha sonra da kullanılabilmesi için kayıt yapmaktadır.
    import pickle
    filename = name+'hosing.sav'
    pickle.dump(model , open(filename, 'wb'))
    
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_validation, Y_validation)
    #print(result)
    """