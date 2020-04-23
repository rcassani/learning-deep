# Deep Forward Networks
A deep forward network or [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a fully connected neural network, i.e., all the nodes from the current layer are connected to the next layer. A **MLP** consisting in 3 or more layers: an input layer, an output layer and one or more hidden layers. Note that the activation function for the nodes in all the layers (except the input layer) is a non-linear function.

<p align="center">
  <img src="https://raw.githubusercontent.com/ledell/sldm4-h2o/master/mlp_network.png" />
</p>

<p align="center">
Example architecture of a MLP. Image [source](https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.Rmd)
</p>

</br>

* The scripts in `mlp-example` present the MLP class (`mlp.py`), which is an numpy didactic implementation, in other words, it's not optimized, though it's well commented.

* The `1_mnist` folder contains the training and testing of a MLP, these actions are performed using as base the [`0_pytorch_template`](../0_pytorch_template)
