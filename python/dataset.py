#! /usr/bin/env python
from csc import *
# Plot Plain Dataset
plotData(X_train, y_train)
savePlot('dataset.png')

# Plot Neyman Pearson Lemma
plotData(X_train, y_train)
plotContour(neyman_pearson_lemma)
savePlot('np_classifier.png')
plotTestStatistic(y_test, np_prediction)
savePlot('np_statistic.png')
plotTestStatistic(y_test, np_prediction, cut=0.5)
savePlot('np_statistic_cut.png')
plotTestStatistic(y_test, np_prediction, cut=0.5)
plt.ylim((0, 300))
savePlot('np_statistic_cut_zoomed.png')

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction))
savePlot('np_roc.png')

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction))
import sklearn.metrics
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, np_prediction)
#plt.gca().text(0.05, 0.3, 'Allowed', fontsize=36)
plt.fill_between(fpr, 0, tpr, color='blue', alpha=0.5)
savePlot('np_roc_shaded.png')

fpr = [1.0]
tpr = [1.0]
for cut in range(1,11):
    figure, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure(figure.number)
    plt.sca(ax1)
    plotTestStatistic(y_test, np_prediction, cut=cut*0.1)
    plt.ylim((0, 300))
    plt.sca(ax2)
    plt.xlabel('False Positive Rate (Type I Error)')
    plt.ylabel('True Positive Rate (1 - Type II Error)')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.0, 1.0])
    fpr.append(((np_prediction > 0.1*cut) & (y_test == 0)).sum() / float((y_test == 0).sum()))
    tpr.append(1.0 - ((np_prediction <= 0.1*cut) & (y_test == 1)).sum() / float((y_test == 1).sum()))
    plt.plot(fpr, tpr, lw=5, label='Receiver operator characteristic')
    figure.subplots_adjust(left=0.05, right=0.95)
    savePlot('np_statistic_cut_{}.png'.format(cut))

x, y = np.mgrid[-5:5:.01, -5:5:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
plt.contourf(x, y, 0.5*(g1.pdf(pos) + g2.pdf(pos)))
savePlot('np_distribution_signal.png')
plt.contourf(x, y, 0.5*(g3.pdf(pos) + g4.pdf(pos)))
savePlot('np_distribution_background.png')

# Do discriminant analysis
import sklearn.lda
lda_classifier = sklearn.lda.LDA()
lda_classifier.fit(X_train, y_train)
lda_prediction = lda_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(lda_classifier.predict_proba)
savePlot('lda_classifier.png')
plotTestStatistic(y_test, lda_prediction)
savePlot('lda_statistic.png')

import sklearn.qda
qda_classifier = sklearn.qda.QDA()
qda_classifier.fit(X_train, y_train)
qda_prediction = qda_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(qda_classifier.predict_proba)
savePlot('qda_classifier.png')
plotTestStatistic(y_test, qda_prediction)
savePlot('qda_statistic.png')

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction), ('LDA', lda_prediction), ('QDA', qda_prediction))
savePlot('lda_qda_roc.png')


