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
print(NPARAMS)
print(NTRAIN)
print(NTEST)


# %%
F = np.array(dataset)[:,:-1]
Y = np.array(dataset)[:,-1]
teta = np.matmul(LA.pinv(F), Y)


# %%
def RS(X, Y, tau):
    return LA.pinv(np.add(X.T.dot(X), np.cov(X.T).dot(tau) )).dot(X.T).dot(Y)


# %%
def RSS(F, Y, sig):
    tau = 1 / sig
    teta = RS(F, Y, tau)
    return LA.norm(F.dot(teta) - Y) ** 2 + (0.5 * LA.norm(teta) ** 2) / sig 


# %%
best = 99999999999999
best_sig = 0
for s in range(991800, 992000, 10):
    res = RSS(F, Y, s)
    if res < best:
        best = res
        best_sig = s


# %%
print(best)
print(best_sig)


# %%
def nrmse(actual, predicted):
    mse = np.mean(np.square(actual - predicted))
    return np.sqrt(mse) / (actual.max() - actual.min())


# %%
def estimate(ds):
    pred = ds.values[::,:-1:]@teta
    return nrmse(pred, ds.values.T[-1])


# %%
print("NRMSE for LS method on train dataset: %s" % estimate(dataset))
print("NRMSE for LS method on test dataset: %s" % estimate(test_dataset))


# %%
def L(predict, real):
    return (predict - real) ** 2


def L_der(predict, real):
    return 2 * (predict - real)


# %%
def init_w(n_param):
    random.seed()
    return [random.uniform(-1/(2*n_param), 1/(2*n_param)) for i in range(n_param)]


# %%
def init_q(X, Y, w):
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
        x, y = X[k], Y[k]
        predict = scalar_product(x, w)
        loss = L(predict, y) + 0.5 * tau * np.linalg.norm(w)
        learning_rate = 0.009 / iterations
        for i in range(features):
            gradient = L_der(predict, y) * x[i]
            w[i] = w[i] * (1 - learning_rate * tau) - learning_rate * gradient
        decay = 0.1
        Q_previous = Q
        Q = (1 - decay) * Q + decay * loss
        if iterations == iterations_limit or (abs(Q - Q_previous) < 0.000001 and with_Q_limit):
            break
        #print(w)
    return w


# %%
def draw_graph(x_values, y_values, x_name, y_name):
    plt.plot(x_values, y_values, 'b')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    #plt.xscale('log')
    plt.show()


# %%
def visual(dataset, I=25):
    vals = []
    for iter in range(I):
        W = SGD(dataset.values[::,:-1:], dataset.values.T[-1], 0.0000010082, 25, False)
        vals.append( nrmse(dataset.values.T[-1], dataset.values[::,:-1:]@(np.array(W)*10e-200)) )
    draw_graph(list(range(I)), vals, "iterations", "nrmse")


# %%
visual(dataset / 2, 100)


# %%
visual(test_dataset / 2, 100)


# %%



