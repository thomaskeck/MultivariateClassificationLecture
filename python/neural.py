#! /usr/bin/env python
from csc import *

import pydot
import subprocess

class NeuralNetwork(object):
    def __init__(self, n_input_neurons, n_hidden_neurons):
        #i = np.random.randint(10000)
        #print(i)
        i = 4079
        np.random.seed(i)
        self.hidden_weights = np.random.normal(size=(n_input_neurons, n_hidden_neurons))
        self.output_weights = np.random.normal(size=n_hidden_neurons)

    def transfer(self, x): 
        return np.tanh(x)
    def dtransfer(self, x): 
        return 4*np.cosh(x)**2/(np.cosh(2*x) + 1)**2

    def fit(self, features, target):
        features  = np.hstack([features, np.ones((len(features), 1))])
        for step in range(1, 1000):
            # Propagte input trough hidden and output layer
            hidden_i = np.dot(features, self.hidden_weights)
            hidden_o = self.transfer(hidden_i)
            output_i = np.dot(hidden_o, self.output_weights)
            output_o = self.transfer(output_i)
            # Back-Propagate error signal
            output_error_signal = self.dtransfer(output_i)*(output_o - target)
            hidden_error_signal = self.dtransfer(hidden_i)*np.outer(output_error_signal, self.output_weights)
            # Update weights
            learn_rate = 0.01 / (1 + int(np.log10(step)))
            self.output_weights -= learn_rate * np.dot(output_error_signal.T, hidden_o)
            self.hidden_weights -= learn_rate * np.dot(features.T, hidden_error_signal)
        print "Error", np.sum((output_o - target)**2)

    def predict_proba(self, features):
        features  = np.hstack([features, np.ones((len(features), 1))])
        # Propagate input trough hidden layer
        hidden_output = self.transfer(np.dot(features, self.hidden_weights))
        # Propagete output of hidden layer through output layer
        p = self.transfer(np.dot(hidden_output, self.output_weights))
	return np.c_[1-p, p]

    @staticmethod
    def export_graphviz(classifier):
	a = 'digraph {\n'
	a += 'splines = line;\n'
	a += 'rankdir=LR;\n'
	a += 'subgraph cluster_input_layer {\n'
        a += ' node [style=filled];\n'
	a += ' label="input layer";\n'
	#a += 'i{i} [label=Bias,style=filled,fillcolor="#CA9EEB"];\n'.format(i=2)
	a += 'i{i} [label=X,style=filled,fillcolor="#9ACEEB"];\n'.format(i=0)
	a += 'i{i} [label=Y,style=filled,fillcolor="#9ACEEB"];\n'.format(i=1)
	a += '\n}\n'
	a += 'subgraph cluster_hidden_layer {\n'
        a += ' node [style=filled];\n'
	a += ' label="hidden layer";\n'
	a += ';'.join(map(lambda x: 'h{i} [label=<H<SUB>{i}</SUB>>,style=filled,fillcolor="#9ACEEB" ]'.format(i=x), range(classifier.hidden_weights.shape[1])))
	a += '\n}\n'
	a += 'subgraph cluster_output_layer {\n'
        a += ' node [style=filled];\n'
	a += ' label="output layer";\n'
	a += ' o [label=O,style=filled,fillcolor="#9ACEEB" ];'
	a += '\n}\n'
	for i, h in enumerate(classifier.hidden_weights):
            if i == 2:
                continue
	    for j, weight in enumerate(h):
	        a += 'i{i} -> h{j}[penwidth={weight:.2f},weight="{weight:.2f}"];\n'.format(i=i, j=j, weight=np.sqrt(abs(weight)))
	for i, weight in enumerate(classifier.output_weights):
	    a += 'h{i} -> o[penwidth={weight:.2f},weight="{weight:.2f}"];\n'.format(i=i, weight=np.sqrt(abs(weight)))
	return a + '\n}'


#import sklearn.neural_network
#mlp_classifier = sklearn.neural_network.MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(4,), random_state=1)

mlp_classifier = NeuralNetwork(3, 3)
mlp_classifier.fit(X_train, y_train)
mlp_prediction = mlp_classifier.predict_proba(X_test)[:, 1]
plotData(X_train, y_train)
plotContour(mlp_classifier.predict_proba)
savePlot('mlp_classifier.png')
plotTestStatistic(y_test, mlp_prediction)
savePlot('mlp_statistic.png')

plotROC(y_test, ('Neyman Pearson Lemma', np_prediction), ('Multi Layer Perceptron', mlp_prediction))
savePlot('mlp_roc.png')

dot_data = NeuralNetwork.export_graphviz(mlp_classifier)
print dot_data
graph = pydot.graph_from_dot_data(dot_data)
graph.write_png('mlp_visualization.png')
subprocess.call("convert -resize 2000x1200 mlp_visualization.png mlp_visualization.png", shell=True)

from scipy.special import expit
sns.set(font_scale=8.0)
X = np.linspace(-6, 6, 100)
plt.plot(X, expit(X), label=None, lw=16)
savePlot('sigmoid.png')
