#!/usr/bin/env python
# coding: utf-8

# In[4]:


def f(x):
    return x**4
def df(x):
    return 4*x
def gradient_descent(x, Lr, iter, df, f):
    print(" \t\t\t Iteration\t\t x\t\t\t\t\t f(x)")
    for l in Lr:
        for i in range(iter):
            x = x - l * df(x)
            print(f"learning rate : {l} \t time: {i}\t\t x: {x}\t\t\t f(x): {f(x)} ")
    return x

a = [ 0.001, 0.01, 0.1, 0.4]
test = gradient_descent(x = 5, Lr = a , iter = 10, df = df , f = f )

