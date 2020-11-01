# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
import random
from matplotlib import pyplot as plt
from scipy.spatial import distance
filename = "datasets/LR/1.txt"


# %%
val = pd.read_csv(filename, nrows=2, header=None)
NPARAMS = val[0][0]
NTRAIN = val[0][1]
dataset = pd.read_csv(filename, sep=' ', skiprows=2, nrows=NTRAIN-1, header=None)
pd.DataFrame(dataset.values)


# %%
test_dataset = pd.read_csv(filename, sep=' ', skiprows=3+NTRAIN, header=None)
NTEST = len(test_dataset)
pd.DataFrame(test_dataset.values)


# %%
print("Number of parameters: %d" % NPARAMS)
print("Number of objects in training dataset: %d" % NTRAIN)
print("Number of objects in test dataset: %d" % NTEST)


# %%
# dataset normalization
def find_minmax_X(X):
    minmax = np.zeros((len(X[0]), 2))
    for i in range(len(X[0])):
        column = X[:, i]
        minmax[i] = [column.min(), column.max()]
    return minmax


def find_minmax_Y(Y):
    return np.array([Y.min(), Y.max()])


def normalize(X, Y, with_constant_feature=True):
    X_normalized, Y_normalized = np.zeros(np.shape(X)), np.zeros(np.shape(Y))
    minmax_X, minmax_Y = find_minmax_X(X), find_minmax_Y(Y)
    for i in range(len(X)):
        Y_normalized[i] = (Y[i] - minmax_Y[0]) / (minmax_Y[1] - minmax_Y[0])
        for j in range(len(X[0])):
            if with_constant_feature and j == 0:
                X_normalized[i][0] = 1
            elif minmax_X[j][0] == minmax_X[j][1]:
                X_normalized[i][j] = 0
            else:
                X_normalized[i][j] = (X[i][j] - minmax_X[j][0]) / (minmax_X[j][1] - minmax_X[j][0])
    return X_normalized, Y_normalized

# %% [markdown]
# **The method of least squares**

# %%
F = np.array(dataset)[:,:-1] # objects matrix
Y = np.array(dataset)[:,-1] # answers vector
teta = np.matmul(LA.pinv(F), Y) # teta vector


# %%
def RS(X, Y, tau): 
    # diffirintiate risk function to get optimal teta 
    return LA.pinv(np.add(X.T.dot(X), np.cov(X.T).dot(tau) )).dot(X.T).dot(Y) # also can use inverse matrix


# %%
def RSS(F, Y, sig):
    # lets add regularization to risk function
    tau = 1 / sig
    teta = RS(F, Y, tau)
    return LA.norm(F.dot(teta) - Y) ** 2 + (0.5 * LA.norm(teta) ** 2) / sig 


# %%
# minimaze empirical risk and finding optimized tau(sigma)
def find_optimal(F, Y):
    best = 99999999999999
    best_sig = 0
    for s in range(991800, 992000, 10):
        res = RSS(F, Y, s)
        if res < best:
            best = res
            best_sig = s
    return best, best_sig


# %%
best, best_sig = find_optimal(*normalize(F, Y))
print("Minimal empirical risk: {} reached with sigma: {} (tau: {}) ".format(best, best_sig, 1/best_sig))

# %% [markdown]
# **Stochastic gradient descent method**

# %%
def nrmse(actual, predicted):
    # normalized root-mean-square error measure
    mse = np.mean(np.square(actual - predicted))
    return np.sqrt(mse) / (actual.max() - actual.min())


# %%
def estimate(ds):
    pred = ds.values[::,:-1:]@teta # classificate
    return nrmse(pred, ds.values.T[-1])


# %%
print("NRMSE for LS method on train dataset: %s" % estimate(dataset))
print("NRMSE for LS method on test dataset: %s" % estimate(test_dataset))


# %%
def L(predict, real):
    # loss func
    return (predict - real) ** 2


def L_der(predict, real):
    # derivative of loss func
    return 2 * (predict - real)


# %%
def init_w(n_param):
    # initialize weight vector
    random.seed()
    return [random.uniform(-1/(2*n_param), 1/(2*n_param)) for i in range(n_param)]


# %%
def init_q(X, Y, w):
    # initialize esimator
    Q = 0
    for i in range(len(X)):
        Q += L(scalar_product(X[i], w), Y[i])
    return Q / len(X)


# %%
def scalar_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


# %%
def SGD(X, Y, tau, iterations_limit=2000, with_Q_limit=True):
    objects, features = len(X), len(X[0])
    w = init_w(features)
    Q = init_q(X, Y, w)
    iterations = 0
    while True:
        iterations += 1
        k = random.randint(0, objects - 1)
        x, y = X[k], Y[k]   # select random object
        predict = scalar_product(x, w)
        loss = L(predict, y) + 0.5 * tau * np.linalg.norm(w) # estimate loss with regularization (tau)
        l_rate = 0.1 / iterations
        for i in range(features):
            gradient = L_der(predict, y) * x[i]
            # w = w(1−hτ)−h∇Li(w) 
            w[i] = w[i] * (1 - l_rate * tau) - l_rate * gradient
        decay = 0.1
        Q_previous = Q
        Q = (1 - decay) * Q + decay * loss
        if iterations == iterations_limit or (abs(Q - Q_previous) < 0.000001 and with_Q_limit):
            break
        #print(w)
    return w


# %%
def draw_graph(x_values, y_values, x_name, y_name, title):
    plt.plot(x_values, y_values, 'b')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    #plt.xscale('log')
    plt.title(title)
    plt.show()


# %%
def visual(X, Y, title):
    vals = []
    max_iter = [i*15 for i in range(1, 80)]
    for iter in max_iter:
        W = SGD(X, Y, 0.0000010082, iter)
        vals.append( nrmse(Y, X@np.array(W)) )
    draw_graph(max_iter, vals, "iterations", "nrmse", title)


# %%
visual(*normalize(F, Y), "Training")


# %%
F_t = np.array(test_dataset)[:,:-1] # objects matrix
Y_t = np.array(test_dataset)[:,-1] # answers vector

visual(*normalize(F_t, Y_t), "Test")


# %%



