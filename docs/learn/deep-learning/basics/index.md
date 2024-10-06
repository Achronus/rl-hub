# Artificial Neural Networks

So far, we've seen algorithms that perform pretty well on small tasks, but quickly become useless in real-world problems because of the complex nature of the environment.

This is where Deep Learning comes into play. To start with, let's look at the most basic of networks - the *Artificial Neural Network (ANN)*.

## What Is a Neural Network?

A Neural Network (NN) is a type of Machine Learning algorithm, often called a model, that is made up of artificial neurons. Its main goal is to **learn patterns** or **relationships** from a set of data and use this information to make predictions for a given task, such as classification or regression.

There are many variants of NNs today, like Convolutional and Recurrent, but the one we will focus on is an *Artificial* one. It's the most basic NN and provides the foundation for any of its variants. You'll often find it called other names too, like *Feed-forward* and *Position-wise*, these all refer to the same thing.

<figure markdown="span">
    ![Artificial Neural Network](../../../assets/imgs/ann.png)
    <figcaption>Figure 1.1. An Artificial Neural Network (ANN) with three layers (image by author)</figcaption>
</figure>

ANNs are made up of multiple neurons (nodes) that are stacked together into layers. Every network has 2 observable layers:

- **An input layer**: representing our input data
- **An output layer**: providing the networks prediction results

In between these layers, we add what are called **hidden layers**, these are what make NNs extremely powerful. We call these *hidden* because we have no direct control over them and don't directly observe them. These are strictly managed by the network and are used to generate its predictions.

Every node in these layers is densely connected with the nodes in the previous and succeeding layers, meaning that every node in one layer is connected with every other node in the layers around it. However, they are not connected to the nodes in the same layer. You can see an example of this in Figure 1.1.

This idea is inspired by the biological neurons inside the human brain and is the reason why networks can learn complex patterns and relationships. Not only that, but full connections allow the network to use various types of data, such as images, text and audio. They are completely universal! But that's not all - as you'll also see later, the mathematical model isn't as complex as you might think.

Okay, let's move on to a high-level understanding of how a neuron works.

## How A Neuron Works

In its simplest form, neurons are a value between a given range, such as $[0, 1]$ or $[0, \infty]$, depending on its activation function. These functions are crucial for providing non-linear behaviour to the model. We'll discuss these in more detail later.

The neuron's value is known as its *activation*. The network uses these *activation* values to understand how important the neuron is for making a new prediction.

<figure markdown="span">
    ![Neuron activations](../../../assets/imgs/activations.png)
    <figcaption>Figure 2.1. Neuron activity example (image by author)</figcaption>
</figure>

When the value is $0$, the neuron is inactive (off), and when it's at its maximum activation (such as $1$), it is fully activated. Full activation means that the pattern used by the neuron is highly relevant to the current prediction. Most of the time, neurons will have some level of activation.

### MNIST Example

<figure markdown="span">
    ![MNIST Samples](../../../assets/imgs/mnist.jpg)
    <figcaption>Figure 2.2. Example of handwritten digits from the MNIST dataset (image by [deeplake](https://datasets.activeloop.ai/docs/ml/datasets/mnist/))</figcaption>
</figure>

To get a better feel for how they work, let's look at a high-level example. Imagine we're trying to classify handwritten digits from the MNIST dataset, which includes digits ranging from 0 to 9. The model's goal is to correctly identify which digit is provided when passed in through the input layer.

In our output layer, we have 10 neurons, each representing the model's prediction for a specific digit. The value of each neuron shows the likelihood (probability) that the input is that digit. 

Let's say we already have a trained network, and we pass in an image of a 9. Our hope would be that each layer would correspond to a different piece of each number. Like in Figure 2.3, where the black nodes are inactive, the white is fully active, and the grey is partially active.

<figure markdown="span">
    ![NN Desired Activations](../../../assets/imgs/active-chain.jpg)
    <figcaption>Figure 2.3. Example of desired NN activations (image by [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi))</figcaption>
</figure>

Logically, this would make sense. But unfortunately, we actually get intangible nonsense - or at least to us anyway 😅! 

<figure markdown="span">
    ![NN Activation Maps](../../../assets/imgs/active-patterns.jpg)
    <figcaption>Figure 2.4. An example of neuron activation maps for some of the nodes in the second layer from Figure 2.3, where blue represents active focus (image by [3Blue1Brown](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2))</figcaption>
</figure>

You may be wondering: "*Why?*" Well, that boils down to the complexity and dimensionality of the data. In Figure 2.4, the activation maps are shown as two-dimensional plots. If they were three-dimensional, things would look a lot different. Anything beyond three-dimensional is extremely hard to visualise. NNs often operate in spaces with hundreds of dimensions, so what they "see" is vastly different to what we can even begin to comprehend. 

Let's talk more about that next.

## Data and Dimensionality

So, what is the term *dimensionality* and how does it relate to our network? 

The best way to explain it is through an example. Consider a spreadsheet of data with four columns: *Credit Score*, *Geography*, *Age*, and *Balance*.

<div style="text-align: center;" markdown>

| CreditScore | Geography | Age      | Balance  |
|:-----------:|:---------:|:--------:|:--------:|
| 619         | France    | 42       | £264.06  |
| 608         | Spain     | 41       | £838.86  |
| $\cdots$    | $\cdots$  | $\cdots$ | $\cdots$ |

</div>

Each one of these columns represents a unique measurable property of information, often referred to as a *feature*, *attribute*, or *variable*. 

If we were to pass this data into our NN, we would have a four-dimensional dataset because each feature contributes to a new dimension in a mathematical space. In practice, we represent our input features by the letter $X$ and express them as a matrix, like so:

$$
X = \begin{bmatrix}
        x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\
        x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\
        \vdots & \vdots & \vdots & \vdots \\
        x_{m,1} & x_{m,2} & x_{m,3} & x_{m,4}
\end{bmatrix} \tag{2.1}
$$

??? info "Geography Column"

    Since the *Geography* column is a categorical variable, we would need to convert it into a numerical representation to feed it into the network properly and plot it effectively. One technique is to use one-hot encoding, or you can simply assign integers to each category. 
    
    We'll leave that for another tutorial!

We define the shape of a matrix by its dimensions *(row, column)*. So, starting from the top-left corner we count *(down, right)*. This is also referred to as an *$m$-by-$n$* matrix - $m$ for the number of rows and $n$ for the number of columns. 

Each *column* is one dimension of the data in the matrix, while each *row* is a separate point in high-dimensional space. We can't visualise this with four dimensions, but we can with two and three, as shown in Figure 3.1.

<figure markdown="span">
    ![2D and 3D plots](../../../assets/imgs/2d-3d-space.jpeg)
    <figcaption>Figure 3.1. Examples of a 2D (left) and 3D (right) space (image by author)</figcaption>
</figure>

The main point of these plots is to illustrate what a mathematical space looks like. The points themselves are completely arbitrary, so don't panic if they seem confusing! We'll talk about the dividers shortly.

Going back to our matrix, if we only had two dimensions, let's say the first two columns (Credit Score and Geography), how would we plot these? Well, we'd follow the basic principles of geometry - take the value at the *horizontal* axis (x-axis) and then the *vertical* axis (y-axis), so a single data point in space would be: 

$$
(x_{1,1}, x_{1,2}) = (619, \text{France}) \tag{2.2}
$$ 

In three dimensions, you'd just extend it to the extra column (Age) and plot it on the z-axis:

$$
(x_{1,1}, x_{1,2}, x_{1,3}) = (619, \text{France}, 42) \tag{2.3}
$$

Then, with higher dimensions, the same behaviour would be repeated. Now, what's really interesting about a network's neurons is how they handle these data points. 

??? info "Decision Boundaries"

 Decision boundaries have different names depending on the dimensionality of your data. Here's a brief summary:

 - In two dimensions, we call it a *line*
 - In three dimensions, a *plane*
 - And, in four or more ($n$-dimensions), a *hyperplane*


See the blue dividers between the two categories? These are called *decision boundaries* and each neuron creates one to separate the data into type $a$ or $b$.

This applies to *all* dimensions, and regardless of how many there are, the boundary will always classify the data between two categories. In other words, a neuron is either *active* or *inactive* based on this classification.

🤯 Amazing, right?! But, wait... if they only perform binary classification, how do they manage to learn such complex patterns? Now's a good time to address its mathematical model - the *perceptron*.

## Perceptron

### Hypothesis

To confirm this hypothesis, let's consider a foundational model in statistical analysis: *linear regression*. 

Bear with me here! In a simple linear regression, we formulate the relationship between an input feature $x$ and a continuous output $y$. We mathematically express it as:

$$
\hat{y} = mx + b \tag{3.1}
$$

Where $m$ is the slope of the line and $b$ is where the line crosses the y-axis (y-intercept). Now, there are a few issues with this. $x$ is only *one* feature and our output value is always continuous. But, what if we were to expand on this idea? 

Let's revisit what we've previously discussed:

1. We need a way to create complex patterns, which requires multiple features
2. We need our output within a range suitable for classification, such as a probability distribution $[0, 1]$

For our first point, what if we added more features, like in *multiple linear regression*? That could work! We just expand our features to $x_n$, but what about the slope? That makes things a lot trickier.

Now that we have multiple features, we're creating a higher dimensional space, so we need a way to understand how each input feature contributes to the model's prediction. Let's try adding a *weight* value to each feature $w_n$ based on it's importance, like so:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b \tag{3.2}
$$

Okay, this solves our first problem, but how do we change it from a continuous output to something we can use for classification? Well, as we're looking for a probability distribution, why don't we use a function to transform our output into a more suitable scale? We can try this with *logistic regression*.

Common for binary classification, it uses a sigmoid function to map its linear values into a range between $[0, 1]$. Formulated:

$$
S(z) = \frac{1}{1 + e^{-z}} \tag{3.3}
$$

Where $z$ is the linear combination of inputs that takes $x_n$ number of features, $w_n$ number of weights, and a bias term $b$ that shifts the decision boundary away from the origin $(0, 0)$ to help classify more complex features:

$$
z = w_1 x_1 + w_2 x_2 + \cdots + b = \sum^{m}_{i = 1}w_i x_i + b \tag{3.4}
$$

Notice how $z$ is identical to $\hat{y}$ in our multiple linear regression.

What if I told you this is exactly what a perceptron is? It's almost too good to be true, right? Something that simple can't possibly be what drives a NN! Well, it is! 😁 Or, at least the first part! Let's take a look.

### Implementation

<figure markdown="span">
    ![Perceptron](../../../assets/imgs/perceptron.png)
    <figcaption>Figure 4.1. A diagram of the perceptron process (image by author)</figcaption>
</figure>



<!-- Update below, factually incorrect -->
In the first layer, each neuron applies a **linear transformation** to the input as different types of linear regressions that split the data into two categories $a$ and $b$. As these neurons outputs are passed to the next layer, their regression lines become slightly more polynomial, shifting into more non-linear shapes like circles, ovals, and spheres. With more layers, this transition continues to happen so that the network can learn more complex patterns that relate to specific parts of our data.

<!-- Add diagrams/animations -->



At it's core, all networks are made up of multiple *perceptrons* (neurons) that are inspired by the human brain.

<!-- Forward Propagation -->

- Activation: a measure of how positive the weighted sum is.
- Weights: what combination of values the neuron picks up on. Measures the strength of the connection between neurons and controls how much each one influences the other.
- Bias: a metric for inactivity. Indicates, how high the weighted sum needs to be before the neuron starts getting meaningfully active. Shifts the activation function enabling it to move in another dimension, modelling more complex functions.
