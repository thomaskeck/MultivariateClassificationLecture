#! /usr/bin/env python
from csc import *

import sklearn.svm

# Linear SVM
linear_svm_classifier = sklearn.svm.SVC(kernel='linear', probability=True)
linear_svm_classifier.fit(X_train, y_train)
linear_svm_prediction = linear_svm_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(linear_svm_classifier.predict_proba)
savePlot('linear_svm_classifier.png')
plotTestStatistic(y_test, linear_svm_prediction)
savePlot('linear_svm_statistic.png')

# rbf SVM
rbf_svm_classifier = sklearn.svm.SVC(kernel='rbf', probability=True)
rbf_svm_classifier.fit(X_train, y_train)
rbf_svm_prediction = rbf_svm_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(rbf_svm_classifier.predict_proba)
savePlot('rbf_svm_classifier.png')
plotTestStatistic(y_test, rbf_svm_prediction)
savePlot('rbf_svm_statistic.png')

# sigmoid SVM
poly_svm_classifier = sklearn.svm.SVC(kernel='poly', probability=True)
poly_svm_classifier.fit(X_train, y_train)
poly_svm_prediction = poly_svm_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(poly_svm_classifier.predict_proba)
savePlot('poly_svm_classifier.png')
plotTestStatistic(y_test, poly_svm_prediction)
savePlot('poly_svm_statistic.png')

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction), ('Linear SVM', linear_svm_prediction), ('RBF SVM', rbf_svm_prediction), ('Poly SVM', poly_svm_prediction))
savePlot('svm_roc.png')
