# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:44:39 2019

@author: 764958
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset as DataFrames and split them into independent and dependent variables
dataset = pd.read_csv('Position_Salaries.csv')

level = dataset.iloc[:, 1:2].values
salary = dataset.iloc[:, -1].values
#The data set is small so no ned for splitting the data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(level,salary)
#for polinomial regression

from sklearn.preprocessing import PolynomialFeatures
#increase or decrease degree to improve the accuracy of the model
poly_set = PolynomialFeatures(degree = 4)
#fit_transform used to train the model and transform the data
level_poly = poly_set.fit_transform(level)
#now we will create a linar regression with the polinomial set
poly_reg = LinearRegression()
poly_reg.fit(level_poly,salary)



#Visulise the simple linear model
plt.scatter(level,salary, color = 'red')
#represent our models finding as a line on the graph
plt.plot(level, lin_reg.predict(level), color = 'blue')
#Lable the graph for better understanding
plt.title('Comapre salary')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()


#for polynomial Pred
plt.scatter(level,salary, color = 'red')
#represent our models finding as a line on the graph
plt.plot(level, poly_reg.predict(poly_set.fit_transform(level)), color = 'blue')
#Lable the graph for better understanding
plt.title('Comapre salary')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()