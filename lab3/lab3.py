#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.datasets import load_iris
import math

iris = load_iris()
X = [[3, 4], [2.5, 4], [1, 4], [2.5, 5], [2, 5], [1.5, 5], [0.5, 5], [1.75, 6], [0.25, 6], [1, 7], [0.25, 7], [0.20, 7], [0.15, 7], [2, 8], [1, 8], [0.15, 8], [0.10, 8], [0.5, 9], [1, 10]]
y = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
 

#filtered_indices = [i for i, target in enumerate(y) if target in target_values]
#X_filtered = X
#y_filtered = [y[i] for i in y]

def sigmoid(z):
    if isinstance(z, list):
        return [1 / (1 + math.exp(-val)) for val in z]
    else:
        return 1 / (1 + math.exp(-z))

def compute_gradients(X, y, w):
    m = len(y)
    z = [sum(X[j][i] * w[i] for i in range(len(w))) for j in range(m)]
    h = sigmoid(z)
    print(f'sigmoid: {sigmoid(z)}')
    dw0 = -sum(y[j] - h[j] for j in range(m)) / m
    dw1 = -sum((y[j] - h[j]) * X[j][0] for j in range(m)) / m
    dw2 = -sum((y[j] - h[j]) * X[j][1] for j in range(m)) / m
    return dw0, dw1, dw2

def gradient_descent(X, y, Lr, epochs):
    w = [0] * len(X[0])
    m = len(y)
    for epoch in range(epochs):
        dw0, dw1, dw2 = compute_gradients(X, y, w)
        w[0] -= Lr * dw0
        w[1] -= Lr * dw1
        w[2] -= Lr * dw2
        
        if epoch % 100 == 0:
            z = [sum(X[j][i] * w[i] for i in range(len(w))) for j in range(m)]
            h = sigmoid(z)
            loss = (-1 / m) * sum(y[j] * math.log(h[j]) + (1 - y[j]) * math.log(1 - h[j]) for j in range(m))
            print(f'Epoch {epoch}: Loss = {loss}')
    return w

learning_rates = [0.0001, 0.001, 0.01]
for lr in learning_rates:
    print(f'Learning Rate: {lr}')
    X_with_ones = [[1] + list(row) for row in X]
    weights = gradient_descent(X_with_ones, y, lr, epochs=1000)
    print(f'Optimized weights: {weights}')



