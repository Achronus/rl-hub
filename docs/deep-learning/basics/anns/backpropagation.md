# Backpropagation

Neural Networks learn using a process called *backpropagation* where the gradient of the loss is propagated backwards through the network using the *Chain Rule* from Calculus.

Here's what the chain rule looks like in Leibniz Notation:

$$
\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}
$$

Where:

- $\frac{dy}{dx}$ is the derivative of $y$ with respect to $x$
- $\frac{dy}{du}$ is the derivative of $y$ with respect to $u$
- $\frac{du}{dx}$ is the derivative of $u$ with respect to $x$

Normally, when working with these expressions you write out the mathematical expression of the derivatives and would manually derive them. In our case, we don't need to that. Our networks are often extremely large so deriving them manually really isn't feasible!

Fortunately, there is a simple way to calculate gradients that we can use for our computations. We'll take a look at this a little later, but first, we need to understand what derivatives and gradients are!

## Derivatives

To start with, let's take a step back and first understand *why* we need derivatives in the first place. 

This mainly boils down to two facts:

1. We are working with high-dimensional data that has complex relationships.
2. Our goal is to minimize a loss function in that high-dimensional space.

Recall that our loss function quantifies the difference between the predicted outputs and the actual target values. You may wonder, if we are measuring the difference between two values, why can't we just *subtract* them?

The problem is, subtraction alone only gives us a static difference between two values at a specific point in space. In other words, it tells us how much one point differs from another. It doesn't tell us *how the loss function behaves as we change the input*.

What we really need is a way to identify the change in *direction and magnitude* of our parameters (weights and biases) in this high-dimensional space so that we can minimize our loss. That's where *derivatives* come in!

> A derivative quantities the *instantaneous rate of change* of a function at a given point.

In our case, it measures how much a change in the parameters affects the output (prediction). This can still be a bit confusing so let's break this down with an example!

Let's say we have an arbitrary function $f(x)$ which performs the following calculation:

$$
f(x) = 3x^2 - 4x + 5
$$

If $x = 3$, then $f(x) = 20$, like so:

$$
f(3) = 3 * 3^2 - 4 * 3 + 5 = 27 - 12 + 5 = 20
$$

Simple enough, right? Now, what if we wanted to identify the rate of change between say $f(3)$ and $f(3.001)$? Such as, checking if the change is positive (up) or negative (down) in our dimensional space?

Well, we can use the derivative! To keep things simple, we will use an approximation of the derivative by computing the *slope* (rise over run) between these two points. We define this as:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

Where:

- $f'(x)$ - is the derivative of $f(x)$ at point $x$. This can also be denoted $\frac{df}{dx}$, indicating the rate of change of $f$ with respect to $x$
- $h$ - is a small change in $x$. The smaller $h$ gets, the closer we get to the true slope of the function at $x$
- $f(x + h)$ - the value of the function $f$ after $x$ has been shifted by $h$
- $f(x)$ - the value of the function $f$ at $x$
- $\lim_{h \to 0}$ - the *limit* of $f$ as the value $h$ approaches $0$

Using our $f(x)$ from before we can approximate our derivative with a small $h$, such as $h = 0.001$:

$$
\frac{df}{dx} \approx \frac{f(3 + 0.001) - f(3)}{0.001} = \frac{20.014003 - 20}{0.001} = \frac{0.014003}{0.001} = 14.003
$$

And there we go! Our derivative of our function with respect to $3$ is $14$.

### Partial Derivatives

So far, we've only looked at the derivative for a single variable, but what if we have multiple?

Let's take a simple equation as an example:

$$
d = ab + c
$$

Notice how $d$ now depends on multiple variables: $a$, $b$, and $c$. With what we've seen so far, there's no way we can calculate it's derivative! Hmmm... what if we break it down into smaller chunks, such as $d$ with respect to $a$? That would be a lot easier to work with, but we still need to figure out how to deal with $b$ and $c$. What if we ignore them entirely, and treat them as constant values of $0$? Now things become a lot simplier!

This is the idea behind *partial derivatives* - we measure the rate of change of a function with respect to one variable while setting the others as constants. Our notion changes a little here to use the $\partial$ symbol instead of $d$.

For example, the partial derivation of $d$ with respect to $a$ is:
$$
\frac{\partial d}{\partial a}
$$

Let's see this in action! We'll start with the partial derivative above, $d$ with respect to $a$:

$$
\frac{\partial{d}}{\partial{a}} = \frac{\partial{(ab)}}{\partial{a}} + 0 = b
$$

We treat $b$ and $c$ as constants and cancel out the $\partial{a}$'s leaving us with $b$. Next, the partial derivative $d$ with respect to $b$:

$$
\frac{\partial{d}}{\partial{b}} = \frac{\partial{(ab)}}{\partial{b}} + 0 = a
$$

This time, we cancel out the $\partial{b}$'s, leaving $c$ as a constant and result in $a$. And lastly, the partial derivative $d$ with respect to $c$:

First, we treat the term $a \cdot b$ as a constant:

$$
\frac{\partial{(ab)}}{\partial{c}} = 0
$$

Then, we get the derivative of $c$ with respect to itself:

$$
\frac{\partial{c}}{\partial{c}} = 1
$$

Put it all together, and we are left with $1$:

$$
\frac{\partial{d}}{\partial{c}} = 0 + 1 = 1
$$

<!-- To complete... -->
Great! Now that we've got the partials ...
