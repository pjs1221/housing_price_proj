# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:26:18 2020

@author: shinp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('Data/eda_data_cleaned.csv')

df_model = df

#Get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


X = df_dum.drop('price',axis=1)
y = df_dum.price.values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso

lm = LinearRegression()
lm.fit(X_train,y_train)

lm_scores = cross_val_score(lm,X_train, y_train,scoring = 'neg_root_mean_squared_error', cv =10)
lm_avg = np.mean(lm_scores)

print('Linear Regression Cross Validation Accuracy Scores:',lm_scores)
print('Linear Regression Cross Validation Accuracy Mean:',lm_avg)

# Lasso Regression

lm_l = Lasso()
lm_l.fit(X_train,y_train)

lm_l_scores = cross_val_score(lm,X_train, y_train,scoring = 'neg_root_mean_squared_error', cv =10)
lm_l_avg = np.mean(lm_scores)


# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train,y_train)

rf_scores = cross_val_score(rf,X_train, y_train,scoring = 'neg_root_mean_squared_error', cv =10)
rf_avg = np.mean(rf_scores)

print('Random Forest Cross Validation Accuracy Scores:',rf_scores)
print('Random Forest Cross Validation Accuracy Mean:',rf_avg)

# Support Vector Machine
from sklearn.svm import SVR

svm = SVR()
svm.fit(X_train,y_train)

svm_scores = cross_val_score(rf,X_train, y_train,scoring = 'neg_root_mean_squared_error', cv =10)
svm_avg = np.mean(svm_scores)

print('SVM Cross Validation Accuracy Scores:',svm_scores)
print('SVM Cross Validation Accuracy Mean:',svm_avg)

#
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [10,20,30,40,50,60,70,80, 90, 100, 110],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1,2,3, 4],
    'min_samples_split': [2,4,6,8],
    'n_estimators': [100, 200, 300, 1000]
}

random_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, cv = 10, n_jobs = -1, verbose = 2)

random_search.fit(X_train,y_train)

random_search.best_params_
best_grid = random_search.best_estimator_


# test ensembles

from sklearn import metrics
tpred_lm = lm.predict(X_test)
tpred_lm_l = lm_l.predict(X_test)
tpred_rf = best_grid.predict(X_test)
tpred_svm = svm.predict(X_test)

lm_accuracy = metrics.mean_squared_error(y_test,tpred_lm,squared=False)
lm_l_accuracy = metrics.mean_squared_error(y_test,tpred_lm_l,squared=False)
rf_accuracy = metrics.mean_squared_error(y_test,tpred_rf,squared=False)
svm_accuracy = metrics.mean_squared_error(y_test,tpred_svm,squared=False)

print('Linear Regression Accuracy:',lm_accuracy)
print('Lasso Regression Accuracy:',lm_l_accuracy)
print('Random Forest Accuracy:',rf_accuracy)
print('SVM Accuracy:',svm_accuracy)

#Plot MAE
sns.distplot(y_test-tpred_lm)
sns.distplot(y_test-tpred_lm_l)
sns.distplot(y_test-tpred_rf)
sns.distplot(y_test-tpred_svm)