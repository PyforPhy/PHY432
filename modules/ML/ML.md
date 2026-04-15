---
layout: default
title: Machine Learning
has_children: true
has_toc: true
nav_exclude: true
---

Machine learning sounds intimidating, but at its core it is something
physicists already know: **fitting a model to data**. If you have ever
fit a line to experimental measurements to extract a physical
parameter, you have already done supervised machine learning.

In **supervised learning** you are given labelled examples — pairs of
inputs and outputs — and you want to learn a function that maps new
inputs to correct outputs. There are two main flavours:

* *Regression* — the output is a continuous number (force, energy,
  band gap). The model is trained by minimising a loss function such
  as the Mean Squared Error (MSE).
* *Classification* — the output is a discrete category (signal vs
  background, ferromagnetic vs paramagnetic phase). The model
  outputs a probability and is trained by minimising a
  cross-entropy loss.

## Gradient Descent

Both regression and classification are solved by **gradient descent**:
start from an initial guess for the model parameters $$\theta$$,
compute how the loss $$\mathcal{L}$$ changes with each parameter, and
take a small step downhill:

$$
\theta \leftarrow \theta - \alpha \, \nabla_\theta \mathcal{L}
$$

where $$\alpha$$ is the *learning rate*. Repeating this update
thousands of times moves the parameters toward a minimum of the loss.

The same algorithm — with the same NumPy building blocks — trains
a simple linear model, a logistic classifier, and a deep neural
network. Only the model architecture and loss function change.

## Neural Networks

A neural network stacks layers of parameterised transformations.
Each hidden neuron computes a weighted sum of its inputs and passes
the result through a non-linear *activation function*. Stacking
multiple such layers lets the network approximate arbitrarily
complex input-output mappings — including curved decision boundaries
that no single straight line could produce.

Training uses **backpropagation**: the chain rule applied layer by
layer, starting from the output loss and working backwards to obtain
gradients for every weight in the network.

## Physics Applications

Machine learning is increasingly used to accelerate materials
discovery and analysis:

* Predicting stable crystal structures and heterointerface geometries
  from density-functional-theory (DFT) data.
* Classifying phases of matter directly from spin configurations
  (Carrasquilla & Melko, *Nature Physics*, 2017).
* Screening thousands of candidate 2D materials for photocatalytic
  properties without running a full DFT calculation for each.
