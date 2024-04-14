#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
data = pd.read_csv('iris.csv')
def gradient_descent(X, y, w1, w0, learning_rate, iterations):
    n = len(y)
    for i in range(iterations):
        y_pred = [w1 * x + w0 for x in X]
        error = [y_pred[j] - y[j] for j in range(n)]
        errorx = [error[j] * X[j] for j in range(n)]

        sum_errorx = sum(errorx)
        sum_error = sum(error)
        
        gradient_w1 = sum_errorx / n
        gradient_w0 = sum_error / n

        w1 = w1 - learning_rate * gradient_w1
        w0 = w0 - learning_rate * gradient_w0
        print(f"Iteration {i+1}: w1 = {w1}, w0 = {w0}") 
        
    return w1, w0


def linear_regression_gradient_descent(data, learning_rate, iterations):
    X = data.iloc[:, 0].values  
    y = data.iloc[:, 3].values  
    w1 = 0 
    w0 = 0

    w1, w0 = gradient_descent(X, y, w1, w0, learning_rate, iterations)
    
    return w1, w0

learning_rate = 0.01
iterations = 100
w1, w0 = linear_regression_gradient_descent(data, learning_rate, iterations)
print(f"Final values: w1 = {w1}, w0 = {w0}")

