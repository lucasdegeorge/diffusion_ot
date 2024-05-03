# Denoising Diffusion Probabilistic Models (DDPms) and Optimal Transport (OT)

This repository explores the intersection between DDPMs and OT theory.

DDPMs are generative models that utilize a forward diffusion process to transform clean data into Gaussian distributions by gradually adding noise. The paper titled *[Understanding DDPM Latent Codes through Optimal Transport](https://arxiv.org/abs/2202.07477)* (Khrulkov 2022) provides insights into this aspect by exploring the relationship between DDPM latent codes and optimal transport theory.

## Theory

The Monge map seeks to minimize transportation costs between probability distributions, while the DDPM encoder map aims to map data distributions to a standard normal distribution. Theoretical results establish that the DDPM encoder map achieves the minimum transportation cost, aligning with the Monge optimal transport map.

## Example Code

The repository contains code snippets and examples demonstrating the theoretical concepts discussed above. It includes implementations of DDPMs, probability flow ODEs, and optimal transport computations using libraries such as [JAX](https://github.com/google/jax), [OTT JAX](https://github.com/ott-jax/ott), and [PyTorch](https://github.com/pytorch/pytorch).
For detailed explanations, please refer to the corresponding sections and files within the repository.
