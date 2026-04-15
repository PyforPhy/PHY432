---
layout: default
title: Neural Networks & Classification
parent: Machine Learning
nav_exclude: true
---

## Learning objectives

By the end of this lesson you should be able to

* distinguish classification from regression and choose appropriate metrics;
* read and interpret a confusion matrix; compute accuracy, precision, and recall;
* explain the precision–recall tradeoff in physical terms (purity vs efficiency);
* describe how logistic regression maps a weighted sum to a probability via the sigmoid;
* explain why a neural network can produce curved decision boundaries while logistic regression cannot;
* describe the role of ReLU and sigmoid activation functions in a two-layer network;
* outline how backpropagation uses the chain rule to compute gradients layer by layer.

## Background

### Classification

In classification the output is a discrete category rather than a
continuous number. In particle physics, for example, we may want to
label each detector event as *signal* or *background*. In condensed
matter we may want to identify whether a spin configuration belongs to
the ferromagnetic or paramagnetic phase.

### Measuring a classifier

Every prediction falls into one of four outcomes:

|  | Predicted: background | Predicted: signal |
|--|--|--|
| **Actual: background** | True Negative (TN) | False Positive (FP) |
| **Actual: signal** | False Negative (FN) | True Positive (TP) |

Three summary statistics:

$$
\text{Accuracy} = \frac{TP + TN}{N}, \qquad
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}
$$

Precision measures **purity** — of the events we selected as signal,
how many actually are? Recall measures **efficiency** — of all real
signal events, how many did we find? Tightening the decision threshold
raises precision but lowers recall; the right tradeoff depends on the
physics goal.

### Logistic regression

The simplest classifier computes a weighted sum of the inputs and
passes it through the *sigmoid* function to obtain a probability:

$$
\hat{p} = \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_2), \qquad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Predict signal if $$\hat{p} \geq 0.5$$. The decision boundary (where
$$\hat{p} = 0.5$$) is always a straight line, so logistic regression
fails on non-linearly separable data such as a ring-shaped signal
cluster.

### Neural networks

A two-layer neural network overcomes this limitation by stacking
transformations:

```
Input (x1, x2)  →  Hidden layer (n neurons, ReLU)  →  Output (sigmoid → probability)
```

Each hidden neuron computes a weighted sum and applies
**ReLU**: $$\max(0, z)$$. If the weighted sum is negative the neuron
is silent; if positive the value passes through unchanged. With many
neurons each activating in a different region of the input space, the
network pieces together a curved boundary from many flat pieces.

The output neuron applies the sigmoid to convert the final value to a
probability between 0 and 1.

Training is still gradient descent. Because the loss depends on the
hidden weights only *indirectly* — through the output layer —
**backpropagation** is needed: apply the chain rule starting from the
output loss and work backwards to obtain gradients for every weight.
