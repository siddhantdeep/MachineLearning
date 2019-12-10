# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:35:37 2019

@author: 764958
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset as DataFrames and split them into independent and dependent variables
dataset = pd.read_csv('50_Startups.csv')

spendings = dataset.iloc[:, :-1].values
profit = dataset.iloc[:,-1:].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([("State", OneHotEncoder(),[3])], remainder="passthrough") 
# The last arg ([0]) is the list of columns you want to transform in this ste
spendings = ct.fit_transform(spendings).astype(float)
#spendings.tolist()
spendings = spendings[:, 1:]

from sklearn.model_selection import train_test_split
spendings_train, spendings_test, profit_train, profit_test = train_test_split(spendings,profit,test_size=0.2,random_state = 0)

#Now we will use the linear_regression model to train a new modekl for multiple regression

from sklearn.linear_model import LinearRegression
multi_reg = LinearRegression()
multi_reg.fit(spendings_train,profit_train)

#now we use the test set of spendings to predict the test set of profits
profit_pred = multi_reg.predict(spendings_test)

#to build an good model we will try to keep the dataset as close to the orignal multiple linear regression equation and apply backward elemination 
import statsmodels.api as sm
sl = 0.05
colm=[0,1,2,3,4,5]
spendings = np.append(arr = np.ones((50,1)).astype(int),values = spendings, axis=1)
spendings_OLS = spendings[:, colm]
spendings_stat = sm.OLS(endog = profit, exog = spendings_OLS).fit()
# To recursively remove the column with the P-value greater than SL
maxP = max(spendings_stat.pvalues).astype(float)
while (maxP>sl):
    for i in range(len(colm)):
        if spendings_stat.pvalues[i]==maxP:
            colm.pop(i)
            spendings_OLS = spendings[:, colm]
            spendings_stat = sm.OLS(endog = profit, exog = spendings_OLS).fit()
            maxP = max(spendings_stat.pvalues).astype(float)
            break
print(colm)
spendings_stat.summary()

 


'''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lab_encoder = LabelEncoder()# TO make the stringgs into numbers
spendings[:, 3] = lab_encoder.fit_transform(spendings[:, 3])
ohe = OneHotEncoder(categorical_features=[3])#Give index of the catagorical field/column and remove number weight 
spendings = ohe.fit_transform(spendings).toarray()
'''
'''
from sklearn.impute import SimpleImputer
var = SimpleImputer(missing_values = np.nan, strategy ='mean')
#Mention the columns where the data is missing
var = var.fit(dataset[:,:])
dataset[:,:] = var.transform(var[:,:])



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])#Give index of the catagorical field/column
independent_variable = ohe.fit_transform(independent_variable).toarray()


#Used to split data based on Logistics
from sklearn.model_selection import train_test_split
years_exp_train, years_exp_test, salary_set_train, salary_set_test = train_test_split(years_exp,salary_set,test_size=1/3,random_state = 0) '''