---
layout: default
title: Regression & Gradient Descent
parent: Machine Learning
nav_exclude: true
---

## Learning objectives

By the end of this lesson you should be able to

* explain what supervised learning is in terms of model fitting;
* implement linear regression using the normal equations in NumPy;
* compute and interpret MSE, MAE, and R² for a fitted model;
* code gradient descent from scratch and use it to train a linear model;
* recognise underfitting and overfitting from training and test error curves.

## Background

### Supervised learning as model fitting

In a physics experiment you collect pairs $$(x_i, y_i)$$ — say,
spring extension and force — and fit a model $$\hat{y} = f(x;\theta)$$
to find the parameters $$\theta$$ (spring constant, offset). This is
exactly supervised learning. The ML terminology maps directly onto
physics language:

| Physics | Machine Learning |
|---------|-----------------|
| Measurements $$(x_i, y_i)$$ | Training data |
| Model $$F = kx + b$$ | Hypothesis |
| Finding best $$k, b$$ | Training |
| Predicting $$F$$ for new $$x$$ | Inference |

### Loss functions

To judge how well a model fits we define a *loss function* that
summarises the residuals $$r_i = y_i - \hat{y}_i$$:

$$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2, \qquad
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|, \qquad
R^2 = 1 - \frac{\sum r_i^2}{\sum (y_i - \bar{y})^2}
$$

MSE is differentiable everywhere and is the standard choice for
gradient-based optimisation.

### Gradient descent

For a linear model there is a closed-form solution (the *normal
equations*). For more complex models — polynomials, neural networks —
no closed form exists, so we use gradient descent instead:

$$
\theta \leftarrow \theta - \alpha \, \frac{\partial \mathcal{L}}{\partial \theta}
$$

Starting from a random initial guess, each update nudges the
parameters slightly in the direction that reduces the loss.

### Overfitting and underfitting

A model that is too simple (e.g. fitting a straight line to a
quadratic signal) has high *bias* — it underfits. A model that is too
complex memorises the training noise and has high *variance* — it
overfits. The test error (evaluated on held-out data) distinguishes
the two cases.
