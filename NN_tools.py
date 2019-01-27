# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def predict_values(model, X, classification=True):
    Wh, bh, Wout, bout = model['Wh'], model['bh'], model['Wout'], model['bout']
    
    net_hidden_layer = np.dot(X,Wh) + bh
    out_hidden_layer = sigmoid(net_hidden_layer)
    net_output_layer = np.dot(out_hidden_layer,Wout) + bout
    out_output_layer = net_output_layer
    if classification:
        out_output_layer = sigmoid(net_output_layer)
    
    return out_output_layer


def datapreprocessing(X,y):
    # formattare correttamente le etichette, ognuna deve essere un array
    # forzo il cast delle etichette ad np.array, se lo è già non succede niente
    y = np.array(y)
    if len(y.shape) == 1:
        y = np.array([[d] for d in y])  
    assert len(y.shape) == 2, "problemi dimensione vettore di etichette"
    
    assert len(X.shape) == 2, "problemi dimensione matrice di input"
    
    #assicurarsi che X e y abbiano stesso numero di righe
    assert X.shape[0] == y.shape[0], 'diverso numero di righe per X e y'
    
    return X, y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    padding = 0.3
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array(predict_values(model,points))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.figure(figsize=(10,7))
    plt.contourf(xx, yy, Z, cmap = plt.get_cmap('Spectral'))
    plt.scatter(X[:, 0], X[:, 1], c=np.array(y).reshape(-1), cmap = plt.get_cmap('Spectral'))
    plt.show()

def plot_loss_accuracy(model, print_accuracy=True):
    plt.figure(figsize=(10,7))
    plt.plot(model["loss_values_train"])
    plt.plot(model["loss_values_valid"],linestyle = '--')
    plt.title('model loss')
    if print_accuracy:
        plt.ylabel('MSE')
    else:
        plt.ylabel('MEE')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='best', fontsize = 15)
    plt.show()

    if (print_accuracy):
        plt.figure(figsize=(10,7))
        plt.plot(model["accuracy_values_train"])
        plt.plot(model["accuracy_values_valid"],linestyle = '--')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='best', fontsize = 15)
        plt.show()



def MEE(y_pred, y_true):
    return np.mean(np.sqrt(np.sum(np.square(y_pred - y_true),axis=1)))


def plot_point_cup(y_pred, y_true):
    plt.figure(figsize=(12,8))
    plt.scatter(y_true.T[0][:], y_true.T[1][:],alpha=0.5, s = 5)
    plt.scatter(y_pred.T[0], y_pred.T[1], alpha= 0.5, s = 10, marker='^')
    plt.legend(['true', 'predicted'], loc='best', fontsize = 15)
    plt.show()



def split_cross_validation(X,y,k,K):
    # validation set: indici di inizio e fine
    ind_start = int(len(X)/K) * (k-1)
    ind_end = int(len(X)/K) * k
    X_valid = np.array(X[ind_start:ind_end])
    X_train = np.delete(X, list(range(ind_start,ind_end)), axis=0)
    y_valid = np.array(y[ind_start:ind_end])
    y_train = np.delete(y, list(range(ind_start,ind_end)), axis=0)
    return X_train,X_valid,y_train,y_valid


def print_grid(line):
    print(("eta: %01.1f, alpha: %01.1f, lambda: %01.4f; loss (valid): %01.6f; loss (train): %01.6f \n") % (line['hyperparam'][0],
    line['hyperparam'][1], line['hyperparam'][2], line['loss_valid'], line['loss_train']))

