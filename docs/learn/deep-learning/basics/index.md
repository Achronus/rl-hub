# Artificial Neural Networks

So far, we've seen algorithms that perform pretty well on small tasks, but quickly become useless in the real-world problems because of the complex nature of the environment.

This is where Deep Learning comes into play. To start with, let's look at the most basic of networks - the *Artificial Neural Network*.

!!! note

    These networks have many names but all mean the samething. Here's some examples:

    - Artificial Neural Networks    
    - Feed-Forward Neural Networks
    - Position-Wise Neural Networks
    - Neural Networks

## Artificial Neural Networks

The most basic form of a Neural Network is an *Arificial* one. This consists of multiple neurons that are put together to create layers which can then learn complex relationships in data and generate a prediction of a new value.

In it's simplest form, neurons are just a value between $[0, 1]$ which represents its *activation*. When $0$, the neuron is turned off, and when $1$ it is fully activated. The network uses these *activation* values to understand how important the neuron's are for creating a new prediction.

To compute the activation, we model the neuron as a *perceptron*. 

## The Perceptron

At it's core, all networks are made up of multiple *perceptrons* (neurons) that are inspired by the human brain.

<!-- Forward Propagation -->

- Activation: a measure of how positive the weighted sum is.
- Weights: what combination of values the neuron picks up on. Measures the strength of the connection between neurons and controls how much each one influences the other.
- Bias: a metric for inactivity. Indicates, how high the weighted sum needs to be before the neuron starts getting meaningfully active. Shifts the activation function enabling it to move in another dimension, modelling more complex functions.
