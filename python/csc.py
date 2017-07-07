#! /usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


sns.set(font_scale=4.0)
sns.set_style("white")

def savePlot(filename):
    figure = plt.gcf()
    figure.set_size_inches(20, 12)
    plt.savefig(filename, transparent=True)
    plt.clf()

def plotData(X, y):
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', s=100, color='orange', label='Signal')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='+', linewidths=3, s=100, color='blue', label='Background')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((-4, 4))
    plt.ylim((-4, 4))
    sns.despine()
    plt.legend(loc='lower left')


def plotTestStatistic(y, p, cut=None):
    _, bins = np.histogram(p, bins=50)
    plt.hist(p[y == 1], color='orange', label='Signal', alpha=0.5, bins=bins)
    plt.hist(p[y == 0], color='blue', label='Background', alpha=0.5, bins=bins)
    if cut is not None:
        plt.gca().text(0.6, 1500, 'Signal (Classifier)', fontsize=36)
        plt.gca().text(0.1, 1500, 'Background (Classifier)', fontsize=36)
        plt.axvline(cut, lw=5, color='black')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.hist(p[(y == 0) & (p > cut)], color='blue', hatch='/', label='Type I Error', alpha=1.0, bins=bins)
        plt.hist(p[(y == 1) & (p <= cut)], color='orange', hatch='\\', label='Type II Error', alpha=1.0, bins=bins)
    plt.xlabel('classifier output')
    plt.ylabel('N')
    sns.despine()
    plt.legend(loc='upper right')


def plotContour(response):
    support = np.arange(-5.0, 5.0, 0.05)
    R = np.array([[float(response(np.array([[x,y]]))[:, 1]) for x in support] for y in support])
    X, Y = np.meshgrid(support, support)
    return plt.contourf(X, Y, R ,alpha=0.5)


import sklearn.metrics
def plotROC(truth, *predictions):
    plt.xlabel('False Positive Rate (Type I Error)')
    plt.ylabel('True Positive Rate (1 - Type II Error)')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.0])
    legend = []
    colors = ['black', 'blue', 'green', 'orange']
    style = ['-', '--', '-.', ':']
    for i, (label, prediction) in enumerate(predictions):
        fpr, tpr, _ = sklearn.metrics.roc_curve(truth, prediction)
        auc = sklearn.metrics.auc(fpr, tpr)
        _label = label + ' AUC = %.3f' % float(auc)
        line, = plt.plot(fpr, tpr, lw=5, color=colors[i], label=_label)
        legend.append((auc, line, _label))
    legend = sorted(legend, key=lambda x: x[0], reverse=True)
    plt.legend(map(lambda x: x[1], legend), map(lambda x: x[2], legend), loc='lower right', fancybox=True)


import sklearn.datasets
#X, y = sklearn.datasets.make_classification(10000, 2, 2, 0, n_clusters_per_class=2, random_state=91, shuffle=False)
X, y = sklearn.datasets.make_classification(10000, 2, 2, 0, n_clusters_per_class=2, random_state=937908, shuffle=False)
X[:, [0,1]] = X[:, [1,0]]

print(X, y)

import scipy.stats
g1 = scipy.stats.multivariate_normal(X[:2500].mean(axis=0), np.cov(X[:2500].T))
g2 = scipy.stats.multivariate_normal(X[5000:7500].mean(axis=0), np.cov(X[5000:7500].T))
g3 = scipy.stats.multivariate_normal(X[2500:5000].mean(axis=0), np.cov(X[2500:5000].T))
g4 = scipy.stats.multivariate_normal(X[7500:10000].mean(axis=0), np.cov(X[7500:10000].T))
def neyman_pearson_lemma(x):
    x = np.atleast_2d(x)
    a = np.exp(-(g1.pdf(x) + g2.pdf(x))/(g3.pdf(x) + g4.pdf(x)))
    return np.c_[1-a, a]

#X, y = sklearn.datasets.make_classification(10000, 2, 2, 0, n_clusters_per_class=2, random_state=91, shuffle=True)
#i = np.random.randint(1000000)
#print i
X, y = sklearn.datasets.make_classification(10000, 2, 2, 0, n_clusters_per_class=2, random_state=937908, shuffle=True)
X[:, [0,1]] = X[:, [1,0]]
X_train, y_train = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

np_prediction = neyman_pearson_lemma(X_test)[:, 1]

if __name__ == '__main__':
    plotData(X_train, y_train)
    plt.show()
