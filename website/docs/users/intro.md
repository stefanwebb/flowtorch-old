---
id: introduction
title: Introduction
sidebar_label: Introduction
slug: /users
---

## What are Normalizing Flows?
*Simply put, a normalizing flow is a learnable function that inputs samples from a simple random distribution, typically Gaussian noise, and outputs samples from a more complex target distribution.*

Mathematically, we represent this as `y = f_\theta(x)` where `x` is a sample of standard Gaussian noise we want `y` to be close to a sample from a target distribution. The function `f`

For instance, [a normalizing flow can be trained](https://arxiv.org/abs/1906.04032) to transform high-dimensional standard Gaussian noise (illustrated conceptually in two dimensions in the table) into samples of a distribution based on a picture of Claude Shannon:

We believe, although still a nascent field, that Normalizing Flows are a fundamental component of the modern Bayesian statistics and probabilistic computing toolkit, and they have already found state-of-the-art applications in Bayesian inference, speech synthesis, and ???, to name a few.

The methods have been applied to such diverse applications as image modeling, text-to-speech, unsupervised language induction, data compression, and modeling molecular structures. As probability distributions are the most fundamental component of probabilistic modeling we will likely see many more exciting state-of-the-art applications in the near future.

The field of normalizing flows can be seen as a modern take on the [change of variables method for random distributions](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function), where the transformations are high-dimensional, often neural networks, and are designed for effective stochastic optimization.

## What is FlowTorch?
[FlowTorch](https://flowtorch.ai) is a library that provides PyTorch components for constructing such flows using the latest research in the field. *Moreover, it defines a well-designed interface so that researchers can easily contribute their own implementations.*

[What is the motivation behind FlowTorch?]

[What are the goals of FlowTorch?]

## Where to From Here?
We recommend reading the next two sections to [install FlowTorch](/users/installation) and [train your first Normalizing Flow](users/start).  For more theoretical background on normalizing flows and information about their applications, see the primer [here](/users/univariate) and the list of survey papers [here](/dev/bibliography#surveys).
