# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:25:35 2019

@author: 764958
"""
#Basic packages for DataScience
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset as DataFrames and split them into independent and dependent variables
dataset = pd.read_csv('Salary_Data.csv')
years_exp = dataset.iloc[:, :-1]
salary_set = dataset.iloc[:,-1:]

#Split Data into Training set(used for training the ML model) and Test Set(Used to compare if the model is working as expected )
from sklearn.model_selection import train_test_split
years_exp_train, years_exp_test, salary_set_train, salary_set_test = train_test_split(years_exp,salary_set,test_size=1/3,random_state = 0)



# feature scaling module is need to normalize certain valuues so that they do not diminish the value of other values
"""
#Not always necessary

from sklearn.preprocessing import StandarScaler
sc_Independent = StandardScaler()
Independent_train = sc_Independent.fit_transform(Independent_train)
Independent_test = sc_Independent.transform(Independent_test)

sc_Dependent = StandardScaler()
Dependent_train = sc_Dependent.fit_transform(Dependent_train)
"""
#This is use to import the type of model you want to use on the data set
from sklearn.linear_model import LinearRegression
regressor_model = LinearRegression()
#Here we "Teach" the model the corelation using the Training values
regressor_model.fit(years_exp_train,salary_set_train)


#Now we use the model to make prediction on the test data present in years_exp_test
salary_set_prediction = regressor_model.predict(years_exp_test)

#Visulising the Results so that it can be easily compared to the actual values 
#THE FOLLOWIN GRAPH IS FOR TRAINING SET ONLY
#We will use this to plot actuall values which we have on the graph as Red Dots
plt.scatter(years_exp_train,salary_set_train, color = 'red')
#represent our models finding as a line on the graph
plt.plot(years_exp_train, regressor_model.predict(years_exp_train), color = 'blue')
#Lable the graph for better understanding
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience (Training set)')
plt.ylabel('Salary')
plt.show()
#THE FOLLOWIN GRAPH IS FOR TEST SET ONLY
plt.scatter(years_exp_test,salary_set_test, color = 'red')
plt.plot(years_exp_train, regressor_model.predict(years_exp_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

