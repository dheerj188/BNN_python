# BNN_python

# Software Simulation of a binarized neural network to implement on Xilinx PyNQ Board.

Algorithm implemented can be found here: https://arxiv.org/pdf/1602.02830

Note that this implementation was carried out to understand BNN functionality, hence all the routines have been implemented from scratch. please use pytorch/ONNX/MxNet for model definition and deployment for real time use cases. 

The training function for MNIST is defined as well. refer to the MNIST_BNN code. 
 
Clip, pop count functions are substituted for accelerated computing and Straight Through Estimator (STE) algorithm is used for backward pass. (refer to the BNN architecture code,)
