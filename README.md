# Multivariate Classification Lecture for the CERN School of Computing

The lecture explains algorithms and concepts used in multivariate classification.
Each chapter explains a specific algorithm and an associated idea or concept.
Most parts apply to machine learning in general.

The lecture intends to engage the interest of students, it is not to be used a stand-alone
introduction into the topic without verbal explanations.

## Content
The content of the lecture is roughly separated into two parts.
The first part focuses on traditional methods used for multivariate classification
in High Energy Physics (excluding neural networks).
The second part introduces neural networks and recent developments
in the field of Deep Learning. The material does not focus on HEP and
summarizes interesting ideas from other fields like image recognition.

### Traditional Methods
  1. Neyman-Pearson Lemma & Supervised Learning
  2. Discriminant Analysis & Analytical Solutions
  3. Decision Trees & Model Complexity
  4. Boosted Decision Trees & Ensemble Methods
  5. Support Vector Machines & Kernel Trick
  6. sPlot & Data Driven Techniques

### Deep Learning
  7. Artificial Neural Networks & Deep Learning
  8. Convolutional Neural Networks & Representation Learning
  9. Recurrent Neural Networks & Sequential Data Processing
  10. Relation Neural Networks & Network Architetures
  11. Adversarial Networks & Generative Models
  12. Playing Games & Reinforcement Learning
  13. Conclusion
  14. Backup


## Further Reading

### General Machine Learning 

Christopher M. Bishop. Pattern Recognition and Machine Learning

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning.

J. Han, M. Kamber, J. Pei. Data Mining: Concepts and Techniques 

**Focused on HEP**

O. Behnke, K. Kröninger, G. Scott, T. Schörner-Sadenius. Data Analysis in High Energy Physics: A Practical Guide to Statistical Methods

### Boosted Decision Trees

Boosted decision trees are the working horse of classification / regression in HEP. They have a good out-of-the-box performance, are reasonable fast, and robust

**Original papers**

Jerome H. Friedman. „Stochastic gradient boosting“ http://statweb.stanford.edu/~jhf/ftp/stobst.pdf

Jerome H. Friedman. „Greedy Function Approximation: A Gradient Boosting Machine“ http://statweb.stanford.edu/~jhf/ftp/trebst.pdf

**uGBoost**

Boosting to uniformity allows to enforce a uniform selection efficiency of the classifier for a certain variable to leave it untouched for a fit

Justin Stevens, Mike Williams "uBoost: A boosting method for producing uniform selection efficiencies from multivariate classifiers" https://arxiv.org/abs/1305.7248

Alex Rogozhnikov et al. „New approaches for boosting to uniformity“. http://iopscience.iop.org/article/10.1088/1748-0221/10/03/T03002/meta

### Deep Learning (Neural Networks)

Deep Learning is the current revolution ongoing in the field of machine learning. Everything from self-driving cars, speech recognition and playing Go can be accomplished using Deep Learning. There is a lot of research going on in HEP, howto take advantage of Deep Learning in our analysis. 

**Standard textbook**

I. Goodfellow, Y. Bengio, A. Courville. Deep Learning (Adaptive Computation and Machine Learning) available online http://www.deeplearningbook.org/

**First paper on usage in HEP (to my knowledge)**

Pierre Baldi, Peter Sadowski, and Daniel Whiteson. „Searching for Exotic Particles in High-Energy Physics with Deep Learning“ https://arxiv.org/abs/1402.4735

**Why does Deep Learning work?**

Henry W. Lin, Max Tegmark, and David Rolnick. Why does deep and cheap learning work so well? https://arxiv.org/abs/1608.08225

**Famous papers by the founding fathers of Deep Learning**

Yann Lecun, Yoshua Bengio, and Geoffrey Hinton. „Deep learning“. https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf

Yoshua Bengio, Aaron C. Courville, and Pascal Vincent. „Unsupervised Feature Learning and Deep Learning: A Review and New Perspectives“. https://arxiv.org/abs/1206.5538

**Adversarial Networks:**

Adversarial networks allow to prevent that a neural networks uses a certain information in its prediction

Gilles Louppe, Michael Kagan, and Kyle Cranmer. „Learning to Pivot with Adversarial Networks“. https://arxiv.org/abs/1611.01046

**Hyperparameter Optimization**

All multivariate methods have hyper-parameters, so some parameters which influence the performance of the algorithm and have to be set by the user. It is common to automatically optimize these hyper-parmaeters using different optimization algorithms. There are four different approaches: grid-search, random-search, gradient, bayesian

**Random search**

James Bergstra and Yoshua Bengio. „Random Search for Hyper-parameter Optimization“ http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

**Gradient-based**

Dougal Maclaurin, David Duvenaud, and Ryan Adams. „Gradient-based Hyperparameter Optimization through Reversible Learning“. http://jmlr.org/proceedings/papers/v37/maclaurin15.pdf

**Bayesian**

Jasper Snoek, Hugo Larochelle, and Ryan P Adams. „Practical Bayesian Optimization of Machine Learning Algorithms“.  http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf

**sPlot**

With sPlot you can train a classifier directly on data, other similar methods are: side-band substration and training data vs mc, both are described in the second paper below

Muriel Pivk and Francois R. Le Diberder. „SPlot: A Statistical tool to unfold data distributions“. https://arxiv.org/abs/physics/0402083

D. Martschei, M. Feindt, S. Honc, and J. Wagner-Kuhr. „Advanced event reweighting using multivariate analysis“. http://iopscience.iop.org/article/10.1088/1742-6596/368/1/012028

### Machine Learning Frameworks


**FastBDT**

https://github.com/thomaskeck/FastBDT

Thomas Keck. „FastBDT: A speed-optimized and cache-friendly implementation of stochastic gradient-boosted decision trees for multivariate classification“. http://arxiv.org/abs/1609.06119.

**TMVA**

http://tmva.sourceforge.net/

Andreas Hoecker et al. „TMVA: Toolkit for Multivariate Data Analysis“. https://arxiv.org/abs/physics/0703039

**FANN**

S. Nissen. Implementation of a Fast Artificial Neural Network Library (fann). http://fann.sourceforge.net/fann.pdf

**SKLearn**

Website http://scikit-learn.org/

F. Pedregosa et al. „Scikit-learn: Machine Learning in Python“. http://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf

**hep_ml**

Website https://arogozhnikov.github.io/hep_ml/

**XGBoost**

Website https://xgboost.readthedocs.io/en/latest/

Tianqi Chen and Carlos Guestrin. „XGBoost: A Scalable Tree Boosting System“. https://arxiv.org/abs/1603.02754

**Tensorflow**

Website https://www.tensorflow.org/

Martin Abadi et al. „TensorFlow: A system for large-scale machine learning“ https://arxiv.org/abs/1605.08695

**Theano**

Website http://deeplearning.net/software/theano/

Rami Al-Rfou et al. „Theano: A Python framework for fast computation of mathematical expressions“ https://arxiv.org/abs/1605.02688

**NeuroBayes**

M. Feindt and U. Kerzel. „The NeuroBayes neural network package“ http://www-ekp.physik.uni-karlsruhe.de/~feindt/acat05-neurobayes
