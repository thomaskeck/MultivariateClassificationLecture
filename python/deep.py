#! /usr/bin/env python
import pydot
import subprocess

architecture = [10, 10, 10, 10, 10, 10, 10, 10, 10, 1]

a = 'digraph {\n'
a += 'splines = line;\n'
a += 'rankdir=LR;\n'

# Layers
for layer, neurons in enumerate(architecture):
    a += 'subgraph cluster_input_layer{layer}'.format(layer=layer)
    a += '{\n'
    a += ' node [style=filled];\n'
    if layer == 0:
        a += ' label="input layer";\n'
    elif layer == len(architecture) - 1:
        a += ' label="output layer";\n'
    else:
        a += ' label="hidden layer {layer}";\n'.format(layer=layer)

    for neuron in range(neurons):
        a += 'l{layer}n{neuron} [label="",style=filled,fillcolor="#9ACEEB"];\n'.format(layer=layer, neuron=neuron)
    a += '\n}\n'

for layer in range(len(architecture)-1):
    for i in range(architecture[layer]):
        for j in range(architecture[layer+1]):
            a += 'l{layer1}n{neuron1} -> l{layer2}n{neuron2}[penwidth={weight:.2f},weight="{weight:.2f}", arrowsize=0.2];\n'.format(layer1=layer, layer2=layer+1, neuron1=i, neuron2=j, weight=1.0)

a += '\n}'

graph = pydot.graph_from_dot_data(a)
graph.write_png('deep_visualization.png')
subprocess.call("convert -resize 2000x1200 deep_visualization.png deep_visualization.png", shell=True)


architecture = [2, 3, 3, 1]
a = 'digraph {\n'
a += 'splines = line;\n'
a += 'rankdir=LR;\n'

# Layers
for layer, neurons in enumerate(architecture):
    a += 'subgraph cluster_input_layer{layer}'.format(layer=layer)
    a += '{\n'
    a += ' node [style=filled];\n'
    if layer == 0:
        a += ' label="input layer";\n'
    elif layer == len(architecture) - 1:
        a += ' label="output layer";\n'
    else:
        a += ' label="hidden layer {layer}";\n'.format(layer=layer)

    for neuron in range(neurons):
        a += 'l{layer}n{neuron} [label="",style=filled,fillcolor="#9ACEEB"];\n'.format(layer=layer, neuron=neuron)
    a += '\n}\n'

for layer in range(len(architecture)-1):
    for i in range(architecture[layer]):
        for j in range(architecture[layer+1]):
            a += 'l{layer1}n{neuron1} -> l{layer2}n{neuron2}[penwidth={weight:.2f},weight="{weight:.2f}", arrowsize=0.2];\n'.format(layer1=layer, layer2=layer+1, neuron1=i, neuron2=j, weight=1.0)

graph = pydot.graph_from_dot_data(a + '\n}')
graph.write_png('rnn_visualization_without.png')
subprocess.call("convert -resize 2000x1200 rnn_visualization_without.png rnn_visualization_without.png", shell=True)

for layer in range(1, len(architecture)-1):
    for i in range(architecture[layer]):
        a += 'l{layer1}n{neuron1} -> l{layer1}n{neuron1}[penwidth={weight:.2f},weight="{weight:.2f}", arrowsize=0.2];\n'.format(layer1=layer, neuron1=i, weight=1.0)

a += '\n}'

graph = pydot.graph_from_dot_data(a)
graph.write_png('rnn_visualization.png')
subprocess.call("convert -resize 2000x1200 rnn_visualization.png rnn_visualization.png", shell=True)

a = 'digraph {\n'
a += 'splines = line;\n'
a += 'subgraph cluster_visible_layer{\n'
a += ' node [style=filled];\n'
a += ' label="visible units";\n'
a += 'vn1 [label="",style=filled,fillcolor="#EB9ACE"];\n'
a += 'vn2 [label="",style=filled,fillcolor="#EB9ACE"];\n'
a += 'vn3 [label="",style=filled,fillcolor="#EB9ACE"];\n'
a += '\n}\n'
a += 'subgraph cluster_hidden_layer{\n'
a += ' node [style=filled];\n'
a += ' label="hidden units";\n'
a += 'hn1 [label="",style=filled,fillcolor="#9ACEEB"];\n'
a += 'hn2 [label="",style=filled,fillcolor="#9ACEEB"];\n'
a += 'hn3 [label="",style=filled,fillcolor="#9ACEEB"];\n'
a += 'hn4 [label="",style=filled,fillcolor="#9ACEEB"];\n'
a += '\n}\n'

for i in range(1, 4):
    for j in range(1, 5):
        a += 'vn{neuron1} -> hn{neuron2}[dir="both",penwidth={weight:.2f},weight="{weight:.2f}", arrowsize=1.0];\n'.format(neuron1=i, neuron2=j, weight=1.0)

a += '\n}'

graph = pydot.graph_from_dot_data(a)
graph.write_png('rbm_visualization.png')
subprocess.call("convert -resize 2000x1200 rbm_visualization.png rbm_visualization.png", shell=True)

from csc import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(font_scale=6.0)
X = np.linspace(-6, 6, 100)
plt.gca().text(-5, 4, 'Rectified linear unit', fontsize=92)
plt.plot(X, np.where(X > 0, X, X*0.2), label=None, lw=8)
savePlot('relu.png')
