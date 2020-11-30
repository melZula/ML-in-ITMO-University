# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score
import cvxopt
from itertools import product


# %%
cvxopt.solvers.options['show_progress'] = False

def replace(data):
    d = {'N': -1, 'P': +1}
    return data.replace({"class": d})

def linear_kernel(x1, x2, *args, **kwargs):
    return np.inner(x1, x2)

def polynomial_kernel(x1, x2, power=2, coef=1, *args, **kwargs):
    return (np.inner(x1, x2) + coef) ** power

def radial_kernel(x1, x2, betha=1, *args, **kwargs):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-betha * distance)


# %%
class SVM:
    def __init__(self, C=1, kernel=radial_kernel, power=4, betha=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.betha = betha
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):
        y = y.astype(np.double)
        n_samples, n_features = np.shape(X)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j], betha=self.betha, power=self.power, coef=self.coef)

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        G_max = np.identity(n_samples) * -1
        G_min = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((G_max, G_min)))
        h_max = cvxopt.matrix(np.zeros(n_samples))
        h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
        h = cvxopt.matrix(np.vstack((h_max, h_min)))

        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        lagr_mult = np.ravel(minimization['x'])
        idx = lagr_mult > 1e-7
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0], self.betha, self.power, self.coef)

    def predict(self, X):
        y_pred = []
        for sample in X:
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample, self.betha, self.power, self.coef)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)


# %%
def draw_plot(clf, dataset_name):
    if dataset_name == 'chips':
        w1, w2 = np.mgrid[min(chips['x']): max(chips['x']):100j,
                    min(chips['y']): max(chips['y']):100j]
    else:
        w1, w2 = np.mgrid[min(geyser['x']): max(geyser['x']):100j,
                    min(geyser['y']): max(geyser['y']):100j]
    label = np.zeros((len(w1), len(w1)))
    for i in range(len(w1)):
        for j in range(len(list(zip(w1, w2))[i][0])):
            label[i][j] = clf.predict([[list(zip(w1, w2))[i][0][j], list(zip(w1, w2))[i][1][j], 1]])[0]
    colors = ['red', 'blue']
    plt.contourf(w1, w2, label, colors=colors, alpha=0.2)
    plt.autoscale(False)
    if dataset_name == 'chips':
        plt.scatter(chips['x'], chips['y'], c=chips['class'], cmap=matplotlib.colors.ListedColormap(colors), zorder=1)
    else:
        plt.scatter(geyser['x'], geyser['y'], c=geyser['class'], cmap=matplotlib.colors.ListedColormap(colors), zorder=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Распределение классов в датасете {dataset_name}')
    plt.show()


# %%
def cross_validation_split(X, y, k_fold=5):
    Xs_train = []
    ys_train = []
    Xs_test = []
    ys_test = []
    n = len(X) // k_fold
    for i in range(k_fold):
        Xs_test.append(X[i * n: (i + 1) * n])
        Xs_train.append(np.concatenate((X[: i * n], X[(i + 1) * n:])))
        ys_test.append(y[i * n: (i + 1) * n])
        ys_train.append(np.concatenate((y[: i * n], y[(i + 1) * n:])))
    return Xs_train, Xs_test, ys_train, ys_test

chips = replace(pd.read_csv('Labs/datasets/chips.csv'))
chips['b'] = np.ones((len(chips), 1))
geyser = replace(pd.read_csv('Labs/datasets/geyser.csv'))
geyser['b'] = np.ones((len(geyser), 1))


# %%
# Чипы

colors = ['red','blue']
plt.scatter(chips['x'], chips['y'], c=chips['class'], cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Распределение классов в датасете chips')
plt.show()


# %%
# Линейное ядро

X = chips.drop(['class'], axis=1).values
y = chips['class'].values
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
params = list(product(C))
accs = []
for param in params:
    clf = SVM(kernel=linear_kernel, C=param[0])
    acc_clf = []
    Xs_train, Xs_test, ys_train, ys_test = cross_validation_split(X, y)
    for i in range(len(Xs_train)):
        clf.fit(Xs_train[i], ys_train[i])
        y_pred = clf.predict(Xs_test[i])
        acc_clf.append(accuracy_score(np.array(y_pred), ys_test[i]))
    accs.append(sum(acc_clf) / len(acc_clf))
print(f'Chips, Линейное ядро, Лучшая точность: {accs[accs.index(max(accs))]}, C: {params[accs.index(max(accs))][0]}')

clf = SVM(kernel=linear_kernel, C=params[accs.index(max(accs))][0])
clf.fit(X, y)
draw_plot(clf, 'chips')


# %%
# Полиномиальное ядро

X = chips.drop(['class'], axis=1).values
y = chips['class'].values
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
powers = [2, 3, 4, 5]
params = list(product(C, powers))
accs = []
for param in params:
    clf = SVM(kernel=polynomial_kernel, C=param[0], power=param[1], coef=2)
    acc_clf = []
    Xs_train, Xs_test, ys_train, ys_test = cross_validation_split(X, y)
    for i in range(len(Xs_train)):
        clf.fit(Xs_train[i], ys_train[i])
        y_pred = clf.predict(Xs_test[i])
        acc_clf.append(accuracy_score(np.array(y_pred), ys_test[i]))
    accs.append(sum(acc_clf) / len(acc_clf))
print(f'Chips, Полиномиальное ядро, Лучшая точность: {accs[accs.index(max(accs))]}, C: {params[accs.index(max(accs))][0]}, power: {params[accs.index(max(accs))][1]}')

clf = SVM(kernel=polynomial_kernel, C=params[accs.index(max(accs))][0], power=params[accs.index(max(accs))][1], coef=1)
clf.fit(X, y)
draw_plot(clf, 'chips')


# %%
# Радиальное ядро

X = chips.drop(['class'], axis=1).values
y = chips['class'].values
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
bethas = [1, 2, 3, 4, 5]
params = list(product(C, bethas))
accs = []
for param in params:
    clf = SVM(C=param[0], betha=param[1])
    acc_clf = []
    Xs_train, Xs_test, ys_train, ys_test = cross_validation_split(X, y)
    for i in range(len(Xs_train)):
        clf.fit(Xs_train[i], ys_train[i])
        y_pred = clf.predict(Xs_test[i])
        acc_clf.append(accuracy_score(np.array(y_pred), ys_test[i]))
    accs.append(sum(acc_clf) / len(acc_clf))
print(f'Chips, Радиальное ядро, Лучшая точность: {accs[accs.index(max(accs))]}, C: {params[accs.index(max(accs))][0]}, betha: {params[accs.index(max(accs))][1]}')

clf = SVM(C=params[accs.index(max(accs))][0],betha=params[accs.index(max(accs))][1])
clf.fit(X, y)
draw_plot(clf, 'chips')


# %%
# Гейзеры

colors = ['red','blue']
plt.scatter(geyser['x'], geyser['y'], c=geyser['class'], cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Распределение классов в датасете geyser')
plt.show()

# Линейное ядро

X = geyser.drop(['class'], axis=1).values
y = geyser['class'].values
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
params = list(product(C))
accs = []
for param in params:
    clf = SVM(kernel=linear_kernel, C=param[0])
    acc_clf = []
    Xs_train, Xs_test, ys_train, ys_test = cross_validation_split(X, y)
    for i in range(len(Xs_train)):
        clf.fit(Xs_train[i], ys_train[i])
        y_pred = clf.predict(Xs_test[i])
        acc_clf.append(accuracy_score(np.array(y_pred), ys_test[i]))
    accs.append(sum(acc_clf) / len(acc_clf))
print(f'Geyser, Линейное ядро, Лучшая точность: {accs[accs.index(max(accs))]}, C: {params[accs.index(max(accs))][0]}')

clf = SVM(kernel=linear_kernel, C=params[accs.index(max(accs))][0])
clf.fit(X, y)
draw_plot(clf, 'geyser')


# %%
# Полиномиальное

X = geyser.drop(['class'], axis=1).values
y = geyser['class'].values
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
powers = [2, 3, 4, 5]
params = list(product(C, powers))
accs = []
for param in params:
    clf = SVM(kernel=polynomial_kernel, C=param[0], power=param[1], coef=2)
    acc_clf = []
    Xs_train, Xs_test, ys_train, ys_test = cross_validation_split(X, y)
    for i in range(len(Xs_train)):
        clf.fit(Xs_train[i], ys_train[i])
        y_pred = clf.predict(Xs_test[i])
        acc_clf.append(accuracy_score(np.array(y_pred), ys_test[i]))
    accs.append(sum(acc_clf) / len(acc_clf))
print(f'Geyser, Полиномиальное ядро, Лучшая точность: {accs[accs.index(max(accs))]}, C: {params[accs.index(max(accs))][0]}, power: {params[accs.index(max(accs))][1]}')

clf = SVM(kernel=polynomial_kernel, C=params[accs.index(max(accs))][0], power=params[accs.index(max(accs))][1], coef=1)
clf.fit(X, y)
draw_plot(clf, 'geyser')


# %%
# Радиальное

X = geyser.drop(['class'], axis=1).values
y = geyser['class'].values
C = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
bethas = [1, 2, 3, 4, 5]
params = list(product(C, bethas))
accs = []
for param in params:
    clf = SVM(C=param[0], betha=param[1])
    acc_clf = []
    Xs_train, Xs_test, ys_train, ys_test = cross_validation_split(X, y)
    for i in range(len(Xs_train)):
        clf.fit(Xs_train[i], ys_train[i])
        y_pred = clf.predict(Xs_test[i])
        acc_clf.append(accuracy_score(np.array(y_pred), ys_test[i]))
    accs.append(sum(acc_clf) / len(acc_clf))
print(f'Geyser, Радиальное ядро, Лучшая точность: {accs[accs.index(max(accs))]}, C: {params[accs.index(max(accs))][0]}, betha: {params[accs.index(max(accs))][1]}')

clf = SVM(C=params[accs.index(max(accs))][0], betha=params[accs.index(max(accs))][1])
clf.fit(X, y)
draw_plot(clf, 'geyser')


