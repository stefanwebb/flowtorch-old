---
id: start
title: Your First Flow
sidebar_label: Your First Flow
---

## The Task
Let's begin learning our first Normalizing Flow with a simple example! The target distribution that we desire to learn will be,
$$
\begin{aligned}
  \tilde{Y} &\sim \mathcal{N}\left(\mu=\begin{bmatrix}
   5 \\
   5
\end{bmatrix}, \Sigma=\begin{bmatrix}
   0.5 & 0 \\
   0 & 0.5
\end{bmatrix} \right)
\end{aligned},
$$
that is, a linear transformation of an standard multivariate normal distribution. The base distribution is,
$$
\begin{aligned}
  X &\sim \mathcal{N}\left(\mu=\begin{bmatrix}
   0 \\
   0
\end{bmatrix}, \Sigma=\begin{bmatrix}
   1 & 0 \\
   0 & 1
\end{bmatrix} \right)
\end{aligned},
$$
that is, standard normal noise (which is typical for Normalizing Flows). The task is to learn some bijection $g_\theta$ so that
$$
\begin{aligned}
  Y &\triangleq g_\theta(X) \\
  &\sim \tilde{Y}
\end{aligned}
$$
approximately holds. We will define our Normalizing Flow, $g_\theta$ by a single affine transformation,
$$
\begin{aligned}
  g_\theta(\mathbf{x}) &\triangleq \begin{bmatrix}
   \mu_1 \\
   \mu_2(x_1)
\end{bmatrix} + \begin{bmatrix}
   \sigma_1 \\
   \sigma_2(x_1)
\end{bmatrix}\otimes\begin{bmatrix}
   x_1 \\
   x_2
\end{bmatrix}.
\end{aligned}
$$
In this notation, $\mathbf{x}=(x_1,x_2)^T$, $\otimes$ denotes element-wise multiplication, and the parameters are the scalars $\mu_1,\sigma_1$ and the parameters of the neural networks $\mu_2(\cdot)$ and $\sigma_2(\cdot)$. (Think of the NNs as very simple shallow feedforward nets in this example.)

There are several metrics we could use to train $Y$ to be close in distribution to $\tilde{Y}$. We will use the KL-divergence between.

In practice, we ... and optimize a Monte Carlo estimate of the KL-divergence with stochastic gradient descent.

*So, to summarize, the task at hand is to learn how to transform standard bivariate normal noise into another bivariate normal distribution using an affine transformation, and we will do so by matching distributions with the KL-divergence metric.*

## Implementation
First, we import the relevant libraries:
```python
import torch
import torch.distributions as dist
import flowtorch
import flowtorch.bijectors as bijectors
```
The base and target distributions are defined using standard PyTorch:
```python
base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
target_dist = dist.Normal(torch.zeros(2)+5, torch.ones(2)*0.5)
```
We can visualize samples from the base and target:
<p align="center">
<img src="/img/bivariate-normal-frame-0.svg" />
</p>

A Normalizing Flow is created in two steps. First, we create a "plan" for the flow as a `flowtorch.Bijector` object,
```python
# Lazily instantiated flow
flow = bijectors.AffineAutoregressive()
```
This plan is then made concrete by calling it with the base distribution,
```python
# Instantiate transformed distribution and parameters
new_dist, params = flow(base_dist)
```
At this point, we have an object, `new_dist`, for the distribution $p_Y(\cdot)$ that follows the standard PyTorch interface. Therefore, it can be trained with the following code, which will be familiar for readers who have used `torch.distributions` before:
```python
# Training loop
opt = torch.optim.Adam(params.parameters(), lr=5e-2)
for idx in range(501):
    opt.zero_grad()
    # Minimize KL(p || q)
    y = target_dist.sample((1000,))
    loss = -new_dist.log_prob(y).mean()
    if idx % 100 == 0:
        print('epoch', idx, 'loss', loss)
        
    loss.backward()
    opt.step()
```
Note how we obtain the learnable parameters of the normalizing flow from the `params` object, which is a `torch.nn.Module`. Visualizing samples after learning, we see that we have been successful in matching the target distribution:
<p align="center">
<img src="/img/bivariate-normal-frame-5.svg" />
</p>
Congratulations on training your first flow!

## Discussion

This simple example illustrates a few important points of FlowTorch's design.

Firstly, [shape info...]

[Sensible defaults...]

[Separation of concerns => bijectors are agnostic to the shapes that they operate on and in many cases are interchangeable with various conditioning networks.]

[Compatibility, in as far as is possible, with `torch.distributions` and `torch.distributions.transform` interfaces.]