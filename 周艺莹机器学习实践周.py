# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:24:41 2021

@author: 周艺莹
"""

import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt  
import statsmodels.api as sm  
import statsmodels.formula.api as smf 
import patsy 


CS=pd.read_csv("cps09mar.csv")
CS.dropna()
CS_Frame= pd.DataFrame(CS, columns = ["age","female","education","earnings",\
                                      "uncov","region","race","marital","hour_wage"])


education_to_level={0:"primary school",\
                    4:"primary school",\
                    6:"primary school",\
                    8:"secondary school",\
                    9:"secondary school",\
                    10:"high school",\
                    11:"high school",\
                    12:"high school",\
                    13:"collage",\
                    14:"collage",\
                    16:"collage",\
                    18:"collage",\
                    20:"collage"}
CS_Frame['educationlevel'] = CS_Frame['education'].map(education_to_level)
marital_to_level={1:"maried",\
                  2:"married",\
                  3:"married",\
                  4:"unmarried",\
                  5:"unmarried",\
                  6:"unmarried",\
                  7:"nevermarried"}
    
CS_Frame=CS_Frame.replace(np.nan,0)

CS_Frame['maritallevel'] = CS_Frame['marital'].map(marital_to_level)    
from sklearn.preprocessing import OneHotEncoder
CS_one_hot=pd.get_dummies(CS_Frame['educationlevel'])
CS_one_hot=pd.get_dummies(CS_Frame['maritallevel'])

X=CS_Frame[["age","female","education","earnings",\
        "uncov","region","race","marital"]]
y=CS_Frame[['hour_wage']]
from sklearn import linear_model
ridge=linear_model.Ridge(alpha=0.5)
ridge.fit(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
X_train_scaled.max(axis=0)))


from sklearn.feature_selection import SelectPercentile
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(CS_Frame), 50))
X_w_noise = np.hstack([CS_Frame, noise])
X_train, X_test, y_train, y_test = train_test_split(
X_w_noise,y, random_state=0, test_size=.5)
select = SelectPercentile(percentile=50)

select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
print("Size of training set: {} size of validation set: {} size of test set:"
" {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score = 0
for alpha in [0.005, 0.05, 0.5, 5, 50, 500]:
    ridge=linear_model.Ridge(alpha=alpha)
scores = cross_val_score(ridge, X_trainval, y_trainval, cv=5)
score = np.mean(scores)
if score > best_score:
    best_score = score
best_parameters = {'alpha': alpha}
ridge=linear_model.Ridge(**best_parameters)
ridge.fit(X_trainval, y_trainval)