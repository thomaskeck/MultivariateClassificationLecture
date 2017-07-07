#! /usr/bin/env python
from csc import *

import sklearn.tree
from sklearn.externals.six import StringIO  

import pydot
import subprocess

# Full Decision Tree
tree_classifier = sklearn.tree.DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)
tree_prediction = tree_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(tree_classifier.predict_proba)
savePlot('tree_classifier.png')
plotTestStatistic(y_test, tree_prediction)
savePlot('tree_statistic.png')

dot_data = StringIO()
sklearn.tree.export_graphviz(tree_classifier, out_file=dot_data, feature_names=['x', 'y'], class_names=['Background', 'Signal'], filled=True, rounded=True, special_characters=True, impurity=False, proportion=True, label='root')
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree_visualization.png')
subprocess.call("convert -resize 2000x1200 tree_visualization.png tree_visualization.png", shell=True)

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction), ('Decision Tree', tree_prediction))
savePlot('tree_roc.png')

# Regularized Decision Tree
reg_tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=3)
reg_tree_classifier.fit(X_train, y_train)
reg_tree_prediction = reg_tree_classifier.predict_proba(X_test)[:, 1]

plotData(X_train, y_train)
plotContour(tree_classifier.predict_proba)
savePlot('reg_tree_classifier.png')
plotTestStatistic(y_test, tree_prediction)
savePlot('reg_tree_statistic.png')

dot_data = StringIO()
sklearn.tree.export_graphviz(reg_tree_classifier, out_file=dot_data, feature_names=['x', 'y'], class_names=['Background', 'Signal'], filled=True, rounded=True, special_characters=True, impurity=False, proportion=True, label='root')
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('reg_tree_visualization.png')
subprocess.call("convert -resize 2000x1200 reg_tree_visualization.png reg_tree_visualization.png", shell=True)

for depth in [1,2,3,4,5,6,7,8, 9]:
    figure, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure(figure.number)
    plt.sca(ax1)
    tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
    tree_classifier.fit(X_train, y_train)
    tree_prediction = tree_classifier.predict_proba(X_test)[:, 1]
    plotData(X_train, y_train)
    plotContour(tree_classifier.predict_proba)
    plt.sca(ax2)
    plotROC(y_test, ('NP', np_prediction), ('Depth = {}'.format(depth), tree_prediction))
    figure.subplots_adjust(left=0.05, right=0.95)
    savePlot('regtree_depth_{}.png'.format(depth))

