"""
This program builds Linear Regression Model for Boston Dataset without using the Python's LinearRegression() function.
This is typically helpful to understand the internal optimization techniques rather than using a go-and-grab code from Python.

"""

import numpy as np 
import random as rd
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# For Stochastic Gradient Descent, this helps to pick samples at random and returns that
def sampleBatches(X, y):
    sample_batch_X = []
    sample_batch_y = []
    
    for i in range(100):
        index = rd.randint(0, len(X)-1)
        sample_batch_X.append(X[index])
        sample_batch_y.append(y[index])
        i += 1
    
    sample_batch_X = np.array(sample_batch_X)
    sample_batch_y = np.array(sample_batch_y)
    
    return sample_batch_X, sample_batch_y


# Finds theta.Xi to get predicted values
def findH(Xi, theta):
    return np.dot(theta.T, Xi.reshape(len(Xi), 1))


# Finds gradient of J(theta)
def gTheta(X, y, theta):
    gTheta = np.dot(np.dot(X.T, X), theta) - np.dot(X.T, y)
    return gTheta
   
    
# Obtains Theta using Gradient Descent algorithm
def gradient_descent(X, y, theta, alpha = 0.00001, n_iter = 100):
    resultTheta = []
    
    for iteration in range(n_iter):
        theta = theta - alpha * gTheta(X, y, theta)
        resultTheta = theta.copy()
        
    return resultTheta
               
    
# Obtains Theta using Stochastic Gradient Descent algorithm
def stochastic_gradient_descent(X, y, theta, alpha = 0.00001, n_iter = 100):
    resultTheta = []
    sample_batch_X, sample_batch_y = sampleBatches(X, y)
    
    for iteration in range(n_iter):
        theta = theta - alpha * gTheta(sample_batch_X, sample_batch_y, theta)
        resultTheta = theta.copy()
    
    return resultTheta


# Obtains Theta using Stochastic Gradient Descent algorithm with momentum
def sgd_momentum(X, y, theta, alpha = 0.00001, n_iter = 100, eta = 0.9):
    resultTheta = []
    sample_batch_X, sample_batch_y = sampleBatches(X, y)
    velocity = 0
    
    for iteration in range(n_iter):
        velocity = eta * velocity - alpha * gTheta(sample_batch_X, sample_batch_y, theta)
        theta = theta + velocity
        resultTheta = theta.copy()
   
    return resultTheta


# Obtains Theta using Stochastic Gradient Descent algorithm with Nesterov momentum
def sgd_nesterov_momentum(X, y, theta, alpha = 0.00001, n_iter = 100, eta = 0.9):
    resultTheta = []
    sample_batch_X, sample_batch_y = sampleBatches(X, y)
    velocity = 0
    
    for iteration in range(n_iter):
        velocity = eta * velocity - alpha * gTheta(sample_batch_X, sample_batch_y, theta + eta * velocity)
        theta = theta + velocity
        resultTheta = theta.copy()
   
    return resultTheta


# Obtains Theta using AdaGrad algorithm
def ada_grad(X, y, theta, alpha = 0.00001, n_iter = 100):
    resultTheta = []
    r = 0
    
    for t in range(n_iter):
        r = r + gTheta(X, y, theta) * gTheta(X, y, theta)
        alpha_t = alpha / np.sqrt(r)
        theta   = theta - alpha_t * gTheta(X, y, theta)
        resultTheta = theta.copy()
   
    return resultTheta


# Obtains Theta using Adam algorithm
def adam(X, y, theta, alpha = 0.00001, n_iter = 100, rho_1 = 0.9, rho_2 = 0.999):
    resultTheta = []
    s = 0 # velocity variable in momentum
    r = 0 # stores exponentially delayed summation of squared gradients
    delta = 1e-8 # constant ~ 0
    
    for t in range(n_iter): # t refers to index of itereation
        s       = rho_1 * s + (1 - rho_1) * gTheta(X, y, theta)   # momentum step 
        r       = rho_2 * r + (1 - rho_2) * gTheta(X, y, theta) * gTheta(X, y, theta) # RMSProp step
        s_hat   = s / (1 - rho_1 ** (t+1))  # bias correction
        r_hat   = r / (1 - rho_2 ** (t+1))  # bias correction
        alpha_t = alpha / np.sqrt(r_hat + delta) # RMSProp step to calculate an adpative learning rate
        v       = -1 * alpha_t * s_hat  # calculate velocity
        theta   = theta + v   # update theta
        resultTheta = theta.copy()
        
    return resultTheta


# main function
if __name__=="__main__":
    # Obtaining dataset from scikit-learn library
    dataset = load_boston()

    # We create a pandas dataframe with columns as 'feature_names' in the dataset and data from 'data' of dataset
    boston = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM']).to_numpy()
    
    # Adds target column and corresponding values to the pandas data frame
    boston['MEDV'] = dataset.target
    y = boston['MEDV'].to_numpy()
    y = y.reshape((y.shape[0], 1))

    # Split data such that: 70% - train, 30% - test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)
    print("X-train: ", X_train.shape, "y-train: ", y_train.shape)
    print("X-test: ", X_test.shape, "y_test: ", y_test.shape)
    
    # Set theta values accordingly between 0 and 1
    thetaList = []
    for i in range(X.shape[1]):
        thetaList.append([0.0000001])
    theta = np.array(thetaList)
    
    # Gradient Descent on Train & Test sets
    gradientDescent_Train = gradient_descent(X_train, y_train, theta)
    gradientDescent_Test = gradient_descent(X_test, y_test, theta)

    # Stochastic Gradient Descent (SGD) on Train & Test sets
    stoc_gradientDescent_Train = stochastic_gradient_descent(X_train, y_train, theta)
    stoc_gradientDescent_Test = stochastic_gradient_descent(X_test, y_test, theta)
    
    # SGD with momentum on Train & Test sets
    SGD_Momentum_Train = sgd_momentum(X_train, y_train, theta)
    SGD_Momentum_Test = sgd_momentum(X_test, y_test, theta)
    
    # SGD with Nesterov momentum on Train & Test sets
    SGD_Nesterov_Train = sgd_nesterov_momentum(X_train, y_train, theta)
    SGD_Nesterov_Test = sgd_nesterov_momentum(X_test, y_test, theta)
    
    # AdaGrad on Train & Test sets
    adaGrad_Train = ada_grad(X_train, y_train, theta)
    adaGrad_Test = ada_grad(X_test, y_test, theta)
    
    # Adam on Train & Test sets
    adam_Train = adam(X_train, y_train, theta)
    adam_Test = adam(X_test, y_test, theta)
    
    # Now we find train & test errors for each algorithm using theta values obtained
    # Gradient Descent algorithm - Train set
    print("\n----- EVALUATION USING GRADIENT DESCENT -----")
    y_pred_for_train = []
    for i in range(len(X_train)):
        y_pred_for_train.append(findH(gradientDescent_Train, X_train[i]))
    y_pred_for_train = np.array(y_pred_for_train)
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_for_train)))
    print("RMSE for train set : ", rmse_train)
    
    # Gradient Descent algorithm - Test set
    y_pred_for_test = []
    for i in range(len(X_test)):
        y_pred_for_test.append(findH(gradientDescent_Test, X_test[i]))
    y_pred_for_test = np.array(y_pred_for_test)
    
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_for_test)))
    print("RMSE for test set : ", rmse_test)
    

    # Stochastic Gradient Descent algorithm - Train set
    print("\n----- EVALUATION USING STOCHASTIC GRADIENT DESCENT -----")
    y_pred_for_train = []
    for i in range(len(X_train)):
        y_pred_for_train.append(findH(stoc_gradientDescent_Train, X_train[i]))
    y_pred_for_train = np.array(y_pred_for_train)
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_for_train)))
    print("RMSE for train set : ", rmse_train)
    
    # Stochastic Gradient Descent algorithm - Test set
    y_pred_for_test = []
    for i in range(len(X_test)):
        y_pred_for_test.append(findH(stoc_gradientDescent_Test, X_test[i]))
    y_pred_for_test = np.array(y_pred_for_test)
    
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_for_test)))
    print("RMSE for test set : ", rmse_test)
    

    # Stochastic Gradient Descent algorithm with momentum - Train set
    print("\n----- EVALUATION USING STOCHASTIC GRADIENT DESCENT WITH MOMENTUM -----")
    y_pred_for_train = []
    for i in range(len(X_train)):
        y_pred_for_train.append(findH(SGD_Momentum_Train, X_train[i]))
    y_pred_for_train = np.array(y_pred_for_train)
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_for_train)))
    print("RMSE for train set : ", rmse_train)
    
    # Stochastic Gradient Descent algorithm with momentum - Test set
    y_pred_for_test = []
    for i in range(len(X_test)):
        y_pred_for_test.append(findH(SGD_Momentum_Test, X_test[i]))
    y_pred_for_test = np.array(y_pred_for_test)
    
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_for_test)))
    print("RMSE for test set : ", rmse_test)
    
    
    # Stochastic Gradient Descent algorithm with Nesterov momentum - Train set
    print("\n----- EVALUATION USING STOCHASTIC GRADIENT DESCENT WITH NESTEROV MOMENTUM -----")
    y_pred_for_train = []
    for i in range(len(X_train)):
        y_pred_for_train.append(findH(SGD_Nesterov_Train, X_train[i]))
    y_pred_for_train = np.array(y_pred_for_train)
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_for_train)))
    print("RMSE for train set : ", rmse_train)
    
    # Stochastic Gradient Descent algorithm with Nesterov momentum - Test set
    y_pred_for_test = []
    for i in range(len(X_test)):
        y_pred_for_test.append(findH(SGD_Nesterov_Test, X_test[i]))
    y_pred_for_test = np.array(y_pred_for_test)
    
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_for_test)))
    print("RMSE for test set : ", rmse_test)
    
    
    # Stochastic Gradient Descent algorithm with Adagrad - Train set
    print("\n----- EVALUATION USING ADAGRAD -----")
    y_pred_for_train = []
    for i in range(len(X_train)):
        y_pred_for_train.append(findH(adaGrad_Train, X_train[i]))
    y_pred_for_train = np.array(y_pred_for_train)
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_for_train)))
    print("RMSE for train set : ", rmse_train)
    
    # Stochastic Gradient Descent algorithm with Adagrad - Test set
    y_pred_for_test = []
    for i in range(len(X_test)):
        y_pred_for_test.append(findH(adaGrad_Test, X_test[i]))
    y_pred_for_test = np.array(y_pred_for_test)
    
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_for_test)))
    print("RMSE for test set : ", rmse_test)
    
    # With Adam algorithm - Train set
    print("\n----- EVALUATION USING ADAM -----")
    y_pred_for_train = []
    for i in range(len(X_train)):
        y_pred_for_train.append(findH(adam_Train, X_train[i]))
    y_pred_for_train = np.array(y_pred_for_train)
    
    rmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_for_train)))
    print("RMSE for train set : ", rmse_train)
    
    # With Adam algorithm - Test set
    y_pred_for_test = []
    for i in range(len(X_test)):
        y_pred_for_test.append(findH(adam_Test, X_test[i]))
    y_pred_for_test = np.array(y_pred_for_test)
    
    rmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_for_test)))
    print("RMSE for test set : ", rmse_test)