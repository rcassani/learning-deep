# Deep Forward Networks
A deep forward network or [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a fully connected neural network, i.e., all the nodes from the current layer are connected to the next layer. A **MLP** consisting in 3 or more layers: an input layer, an output layer and one or more hidden layers. Note that the activation function for the nodes in all the layers (except the input layer) is a non-linear function.

<p align="center">
  <img src="https://raw.githubusercontent.com/ledell/sldm4-h2o/master/mlp_network.png" />
</p>

<p align="center">
Example architecture of a MLP  
Image from [https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.Rmd]
</p>

</br>

1 . The Jupyter notebook [`mlp_notebook.ipynb`](mlp_notebook.ipynb) has as goal to show the use the MLP class [`mlp.py`](mlp.py) provided. The implementation of the MLP has didactic purposes, in other words, it's not optimized, though it's well commented. It's mostly based on the lectures for weeks 4 and 5 (neural networks) in the the MOOC [Machine Learning](https://www.coursera.org/learn/machine-learning#%20) taught by from Andrew Ng and notes from the chapter 6 (deep forward networks) from the [Deep Learning](http://www.deeplearningbook.org/).

2. The Jupyter notebook [`mlp_examples.ipynb`](mlp_examples.ipynb) shows the similarities and differences of diverse implementations of MLP, using the above described `mlp` class, `numpy`, and Pytorch tensors (on CPU and GPU) and Pytorch framework (on CPU and GPU).
