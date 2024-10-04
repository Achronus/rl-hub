# Artificial Neural Networks

So far, we've seen algorithms that perform pretty well on small tasks, but quickly become useless in real-world problems because of the complex nature of the environment.

This is where Deep Learning comes into play. To start with, let's look at the most basic of networks - the *Artificial Neural Network (ANN)*.

## What Is a Neural Network?

A Neural Network (NN) is a type of Machine Learning algorithm, often called a model, that is made up of artificial neurons. It's main goal is to **learn patterns** or **relationships** from a set of data and use this information to make predictions for a given task, such as classification or regression.

There are many variants of NNs today, like Convolutional and Recurrent, but the one we will focus on is an *Artificial* one. It's the most basic NN and provides the foundation for any of its variants. You'll often find it called other names too, like *Feed-forward NNs* and *Position-wise NNs*, these are all the same as *Artificial NNs*.

<figure markdown="span">
    ![Artificial Neural Network](../../../assets/imgs/ann.png)
    <figcaption>Figure 1.1. An Artificial Neural Network (ANN) with three layers (image by author)</figcaption>
</figure>

NNs are made up of multiple neurons (nodes) and are stacked into layers. Every network has at least 2 layers:

- **An input layer**: representing our input data
- **An output layer**: representing the desired output values. For example, for a classification problem between dogs and cats, we would have two output nodes - one for the dog prediction and another for the cat prediction

They become far more powerful when we start to add more layers of neurons in between them called **hidden layers**. We call these *hidden* because we have no direct control over them and don't directly observe them. These are managed solely by the network.

Every node is densely connected with the nodes in the previous and succeeding layers, meaning that every node in one layer is connected with every other node in the layers around it. However, they are not connected to the nodes in the same layer. You can see an example of this in Figure 1.1.

This idea is inspired by biological neurons inside the human brain. 

## How A Neuron Works

In it's simplest form, neurons are just a value between $[0, 1]$ which represents its *activation*. When $0$, the neuron is turned off, and when $1$ it is fully activated. The network uses these *activation* values to understand how important the neuron's are for creating a new prediction.

To compute the activation, we model the neuron as a *perceptron*. 

## The Perceptron

At it's core, all networks are made up of multiple *perceptrons* (neurons) that are inspired by the human brain.

<!-- Forward Propagation -->

- Activation: a measure of how positive the weighted sum is.
- Weights: what combination of values the neuron picks up on. Measures the strength of the connection between neurons and controls how much each one influences the other.
- Bias: a metric for inactivity. Indicates, how high the weighted sum needs to be before the neuron starts getting meaningfully active. Shifts the activation function enabling it to move in another dimension, modelling more complex functions.
