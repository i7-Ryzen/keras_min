# keras_min

## Description
The goal of this project is to make ML models produced through using tensorflow/keras, more portable. Since a trained model's main application is to make predictions, it does not require the whole tensorflow/keras library to run it. In fact, the goal of this project is for you to be able to run your trained model in Python using only Numpy. Obviously, you will need your model.h5 file. We named this project Keras minimizer (or kerasmizer).


## Methodology

The main stages of this project fall under the 3 components:

    1) Code a function that extracts the information needed to build the model.
    2) Code from scratch (numpy / cython) custom layers and activation functions.
    3) Build the deployment model from the extracted information and custom layers

The code is compatible with models using the following functions:

1. **load_model_from_h5** : return a dictionary with the main information (weights and configuration) from an HDF5/h5 file, which is a file format to store structured data. Keras saves models in this format as it can easily store the weights and model configuration in a single file. 
2. **Custom layers** : 
	- **conv2d** :  the 2D convolution layer creates a convolution kernel that is convolved with the layer input to produce an array of outputs. 
	- **Pool2d** : this layer Downsamples the input representation by taking the maximum/average value over the window defined by pool_size for each dimension along the features axis. 
	- **Dense** : It implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.
	- **Flatten** : Flattens the input. 
	- **Zeropadding2D** :This layer can add rows and columns of zeros at the top, bottom, left and right side of an image array.
	- **BatchNormalization** :Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
	- **Add** :It takes as input a list of arrays, all of the same shape, and   returns     a single tensor (also of the same shape). 

 3. **Activations** : relu, sigmoid, softmax, softplus, softsing, tanh …
 4. **run_model** : this function combines the output of the costume layers and the extracted dictionary by making the inference on the layers.


## Results 
In this section, we benchmark the performance of our Keras minimizer using a simple model between using Keras, numpy, cython+cnumpy, and cython.
There are two main reasons to deploy the model with numpy/cython instead of tensorflow:
1. The execution time on the CPU is faster with numpy/cython compared to TensorFlow / keras.
2. Reduce the size of the model storage.

## Challenges

This project involves in understanding the workaround of Neuron Network models. From packaging and unpackaging Keras NN models, we translate a model from Keras to a standalone Python + Numpy code. We tested its accuracy as well as it's efficiency.
This project enhances one understanding of NN functions, instructs one to write codes in different languages including Cython, and to do benchmarking tests.
