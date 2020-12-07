#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Laila Ayman        - 16P3084
#Omar Mostafa Hosny - 16P8170
#Project 2 - CI

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.special


# In[ ]:


#Importing Data
testing_d = pd.read_csv("E:\Cllg\Semester 8\CI\DataSets\MNIST Numbers\mnist_test.csv")    #Import your testing Data here
testing_data = (testing_d.iloc[:,:].values).T

training_d = pd.read_csv("E:\Cllg\Semester 8\CI\DataSets\MNIST Numbers\mnist_train.csv")  #Import your training Data here
training_data = (training_d.iloc[:,:].values).T


# In[ ]:


#Normalizing the Data
fac = 0.99/255

x_train = training_data[1:,:].astype(np.float16) * fac + 0.01
y_train = ((training_data[:1,:]).T).astype(np.int)

x_test = testing_data[1:,:].astype(np.float16) * fac + 0.01
y_test = ((testing_data[:1,:]).T).astype(np.int)


# In[ ]:


#One Hot Labelling
x = np.arange(10)   #Number of Classes is 10

y_train = (x == y_train).astype(np.int)
y_test = (x == y_test).astype(np.int)


# In[ ]:


def initialize_weights(layers_dims):     #Inicilize the weights depending on the number of layers 
    
    parameters = {}
    L = len(layers_dims)     #Number of Layers in the NN
    
    for l in range(1, L):
        
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/(layers_dims[l] + layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters 


# In[ ]:


def linear_forward(A, W, b):
    
    Z = np.dot(W,A) + b       #Implements a Forward linear step
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# In[ ]:


def relu(Z):
    
    A = np.maximum(0,Z)
    
    cache = Z
    
    return A, cache


# In[ ]:


def tanh(Z):
    
    A = np.tanh(Z)
    
    cache = Z
    
    return A, cache


# In[ ]:


def softmax(Z):
    
    A = sp.special.softmax(Z, axis = 0)
    
    cache = Z
    
    return A


# In[ ]:


def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "relu":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation == "tanh":
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
        
    elif activation == "softmax":
        
        linear_cache = (A_prev, W, b)
        A, activation_cache = softmax(A_prev) 
        
    cache = (linear_cache , activation_cache)
        
    return A, cache


# In[ ]:


def l_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2      #Divided 2 because parameters has Ws and bs.
    
    for l in range(1, L):
        
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "relu")
    caches.append(cache)
    
    ALL = sp.special.softmax(AL, axis = 0)
    
    return ALL, caches


# In[ ]:


def calc_cost(AL, Y):
    
    m = Y.shape[0]
    
    cost = 1/m * np.sum(-1 * Y.T * np.log(AL))
    
    cost = np.squeeze(cost)
    
    return cost
    


# In[ ]:


def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ , axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[ ]:


def relu_backward(dA, activation_cache):
    
    Z = activation_cache
    dA[Z<0] = 0
    dZ = dA 
    
    return dZ   


# In[ ]:


def tanh_backward(dA, activation_cache):
    
    Z = activation_cache
    A = np.tanh(Z)
    dZ = (1 - (A)**2) * dA
        
    return dZ


# In[ ]:


def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    if activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db


# In[ ]:


def l_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches)     #Number of Layers
    m = AL.shape[1] 
    Y = Y.T
    
    dAL = (AL - Y)        #Backward Prop against softmax + Cross Entropy Loss
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "relu")
    
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads


# In[ ]:


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


# In[ ]:


def nn_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    
    costs = []
    
    parameters = initialize_weights(layers_dims)
    
    for i in range(0, num_iterations):
        
        AL, caches = l_model_forward(X, parameters)
        
        cost = calc_cost(AL, Y)
        
        grads = l_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 2 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 2 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.legend(('Cost'))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[ ]:


def test_predict(X, Y, parameters):
    
    y_hat, caches = l_model_forward(X, parameters)
    Y = Y.T
    
    accuracy = (np.mean(np.argmax(y_hat, axis = 0) == np.argmax(Y, axis = 0)))
    
    error = 1 - accuracy
    
    accuracy = accuracy * 100
    
    return accuracy, error


# In[ ]:


#For looping on the 1, 2 or 3 Hidden Layers Models with different number of nodes in there hidden layers.


n_x = x_train.shape[0]
n_y = y_train.shape[1]
n_h = 5

test_accuracies = [0]      #To start the graph at 0
train_accuracies = [0]
test_errors = [1]          #To start the graph at 1
train_errors = [1]

for i in range(15):
    layers_dims = [n_x, n_h, n_y]
    layers_dims2 = [n_x, n_h, n_h, n_y]
    layers_dims3 = [n_x, n_h, n_h, n_h, n_y]
    parameters = nn_model(x_train, y_train, layers_dims, learning_rate = 0.15, num_iterations = 300 ,print_cost = True)
    
    test_accuracy, test_error = test_predict(x_test, y_test, parameters)
    train_accuracy, train_error = test_predict(x_train, y_train, parameters)
    
    
    test_accuracies.append(test_accuracy)
    test_errors.append(test_error)
    
    train_accuracies.append(train_accuracy)
    train_errors.append(train_error)
    print(n_h)
    n_h = n_h + 10

print(train_accuracies)
print(test_accuracies)

learning_rate = 0.15
plt.plot(np.squeeze(train_accuracies),  color = 'skyblue', label = "Train Errors")
plt.plot(np.squeeze(test_accuracies), color = 'olive' , linestyle = 'dashed', label = "Test Errors")
plt.legend(('Training','Testing'))
plt.ylabel('accuracies')
plt.xlabel('n_h')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

plt.plot(np.squeeze(train_errors), color = 'skyblue', label = "Train Errors")
plt.plot(np.squeeze(test_errors), color = 'olive' , linestyle = 'dashed', label = "Test Errors")
plt.legend(('Training','Testing'))
plt.ylabel('errors')
plt.xlabel('n_h')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


# In[ ]:


#For testing the No Hidden Layers Models

n_x = x_train.shape[0]
n_y = y_train.shape[1]
layers_dims = [n_x, n_y]

parameters = nn_model(x_train, y_train, layers_dims, learning_rate = 0.3, num_iterations = 300, print_cost = True)

model_accuracy, model_error = test_predict(x_test, y_test, parameters)

print("Testing Accuracy is : ", model_accuracy)
print("Testing Error is : ", model_error)


# In[ ]:




