#! /usr/bin/env python
from csc import *
import sklearn.ensemble

forest_classifier = sklearn.ensemble.GradientBoostingClassifier(subsample=0.5, max_depth=3, n_estimators=50)
forest_classifier.fit(X_train, y_train)
forest_prediction = forest_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(forest_classifier.predict_proba)
savePlot('forest_classifier.png')
plotTestStatistic(y_test, forest_prediction)
savePlot('forest_statistic.png')

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction), ('Gradient Boosted Decision Tree', forest_prediction))
savePlot('forest_roc.png')

