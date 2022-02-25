---
layout: post
comments: true
title: "Evolution Strategies"
date: 2019-09-05 12:00:00
tags: evolution reinforcement-learning
---


> Gradient descent is not the only option when learning optimal model parameters. Evolution Strategies (ES)  works out well in the cases where we don't know the precise analytic form of an objective function or cannot compute the gradients directly. This post dives into several classic ES methods, as well as how ES can be used in deep reinforcement learning.


<!--more-->


Stochastic gradient descent is a universal choice for optimizing deep learning models. However, it is not the only option. With black-box optimization algorithms, you can evaluate a target function $$f(x): \mathbb{R}^n \to \mathbb{R}$$, even when you don't know the precise analytic form of $$f(x)$$ and thus cannot compute gradients or the Hessian matrix. Examples of black-box optimization methods include [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing), [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing) and [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).

**Evolution Strategies (ES)** is one type of black-box optimization algorithms, born in the family of **Evolutionary Algorithms (EA)**. In this post, I would dive into a couple of classic ES methods and introduce a few applications of how ES can play a role in deep reinforcement learning.


{: class="table-of-content"}
* TOC
{:toc}


## What are Evolution Strategies?

Evolution strategies (ES) belong to the big family of evolutionary algorithms. The optimization targets of ES are vectors of real numbers, $$x \in \mathbb{R}^n$$. 

Evolutionary algorithms refer to a division of population-based optimization algorithms inspired by *natural selection*. Natural selection believes that individuals with traits beneficial to their survival can live through generations and pass down the good characteristics to the next generation. Evolution happens by the selection process gradually and the population grows better adapted to the environment.


![EA]({{ '/assets/images/EA-illustration.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. How natural selection works. (Image source: Khan Academy: [Darwin, evolution, & natural selection](https://www.khanacademy.org/science/biology/her/evolution-and-natural-selection/a/darwin-evolution-natural-selection))*

Evolutionary algorithms can be summarized in the following [format](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/06-blackBoxOpt.pdf) as a general optimization solution:

Let's say we want to optimize a function $$f(x)$$ and we are not able to compute gradients directly. But we still can evaluate $$f(x)$$ given any $$x$$ and the result is deterministic. Our belief in the probability distribution over $$x$$ as a good solution to $$f(x)$$ optimization is $$p_\theta(x)$$, parameterized by $$\theta$$. The goal is to find an optimal configuration of $$\theta$$.

> Here given a fixed format of distribution (i.e. Gaussian), the parameter $$\theta$$ carries  the knowledge about the best solutions and is being iteratively updated across generations.


Starting with an initial value of $$\theta$$, we can continuously update $$\theta$$ by looping three steps as follows:
1. Generate a population of samples $$D = \{(x_i, f(x_i)\}$$ where $$x_i \sim p_\theta(x)$$. 
2. Evaluate the "fitness" of samples in $$D$$.
3. Select the best subset of individuals and use them to update $$\theta$$, generally based on fitness or rank.

In **Genetic Algorithms (GA)**, another popular subcategory of EA, $$x$$ is a sequence of binary codes, $$x \in \{0, 1\}^n$$. While in ES, $$x$$ is just a vector of real numbers, $$x \in \mathbb{R}^n$$.



## Simple Gaussian Evolution Strategies

[This](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) is the most basic and canonical version of evolution strategies. It models $$p_\theta(x)$$ as a $$n$$-dimensional isotropic Gaussian distribution, in which $$\theta$$ only tracks the mean $$\mu$$ and standard deviation $$\sigma$$.

$$
\theta = (\mu, \sigma),\;p_\theta(x) \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 I) = \mu + \sigma \mathcal{N}(0, I)
$$

The process of Simple-Gaussian-ES, given $$x \in \mathcal{R}^n$$:
1. Initialize $$\theta = \theta^{(0)}$$ and the generation counter $$t=0$$
2. Generate the offspring population of size $$\Lambda$$ by sampling from the Gaussian distribution:<br/><br/>$$D^{(t+1)}=\{ x^{(t+1)}_i \mid x^{(t+1)}_i = \mu^{(t)} + \sigma^{(t)} y^{(t+1)}_i \text{ where } y^{(t+1)}_i \sim \mathcal{N}(x \vert 0, \mathbf{I}),\;i = 1, \dots, \Lambda\}$$<br/>.
3. Select a top subset of $$\lambda$$ samples with optimal $$f(x_i)$$ and this subset is called **elite** set. Without loss of generality, we may consider the first $$k$$ samples in $$D^{(t+1)}$$ to belong to the elite group --- Let's label them as<br/><br/>$$D^{(t+1)}_\text{elite} = \{x^{(t+1)}_i \mid x^{(t+1)}_i \in D^{(t+1)}, i=1,\dots, \lambda, \lambda\leq \Lambda\}$$<br/>. 
4. Then we estimate the new mean and std for the next generation using the elite set:<br/><br/>
$$
\begin{aligned}
\mu^{(t+1)} &= \text{avg}(D^{(t+1)}_\text{elite}) = \frac{1}{\lambda}\sum_{i=1}^\lambda x_i^{(t+1)} \\
{\sigma^{(t+1)}}^2 &= \text{var}(D^{(t+1)}_\text{elite}) = \frac{1}{\lambda}\sum_{i=1}^\lambda (x_i^{(t+1)} -\mu^{(t)})^2
\end{aligned}
$$<br/>
5. Repeat steps (2)-(4) until the result is good enough ✌️



## Covariance Matrix Adaptation Evolution Strategies (CMA-ES)

The standard deviation $$\sigma$$ accounts for the level of exploration: the larger $$\sigma$$ the bigger search space we can sample our offspring population. In [vanilla ES](#simple-gaussian-evolution-strategies), $$\sigma^{(t+1)}$$ is highly correlated with $$\sigma^{(t)}$$, so the algorithm is not able to rapidly adjust the exploration space when needed (i.e. when the confidence level changes).

[**CMA-ES**](https://en.wikipedia.org/wiki/CMA-ES), short for *"Covariance Matrix Adaptation Evolution Strategy"*, fixes the problem by tracking pairwise dependencies between the samples in the distribution with a covariance matrix $$C$$. The new distribution parameter becomes:


$$
\theta = (\mu, \sigma, C),\; p_\theta(x) \sim \mathcal{N}(\mu, \sigma^2 C) \sim \mu + \sigma \mathcal{N}(0, C)
$$

where $$\sigma$$ controls for the overall scale of the distribution,  often known as *step size*. 

Before we dig into how the parameters are updated in CMA-ES, it is better to review how the covariance matrix works in the multivariate Gaussian distribution first. As a real symmetric matrix, the covariance matrix $$C$$ has the following nice features (See [proof](http://s3.amazonaws.com/mitsloan-php/wp-faculty/sites/30/2016/12/15032137/Symmetric-Matrices-and-Eigendecomposition.pdf) & [proof](http://control.ucsd.edu/mauricio/courses/mae280a/lecture11.pdf)):
- It is always diagonalizable.
- Always positive semi-definite.
- All of its eigenvalues are real non-negative numbers. 
- All of its eigenvectors are orthogonal.
- There is an orthonormal basis of $$\mathbb{R}^n$$ consisting of its eigenvectors.

Let the matrix $$C$$ have an *orthonormal* basis of eigenvectors $$B = [b_1, \dots, b_n]$$, with corresponding eigenvalues $$\lambda_1^2, \dots, \lambda_n^2$$. Let $$D=\text{diag}(\lambda_1, \dots, \lambda_n)$$.


$$
C = B^\top D^2 B
= \begin{bmatrix} 
\mid & \mid &  & \mid \\
b_1 & b_2 & \dots & b_n\\
\mid & \mid &  & \mid \\
\end{bmatrix}
\begin{bmatrix}
\lambda_1^2 & 0 & \dots & 0 \\
0 & \lambda_2^2 & \dots & 0 \\
\vdots & \dots & \ddots & \vdots \\
0 & \dots & 0 & \lambda_n^2
\end{bmatrix}
\begin{bmatrix} 
- & b_1 & - \\
- & b_2 & - \\
  & \dots & \\
- & b_n & - \\
\end{bmatrix}
$$

The square root of $$C$$ is:

$$
C^{\frac{1}{2}} = B^\top D B
$$


{: class="info"}
| Symbol | Meaning |
| ---------- | ---------- |
| $$x_i^{(t)} \in \mathbb{R}^n$$ | the $$i$$-th samples at the generation (t) |
| $$y_i^{(t)} \in \mathbb{R}^n$$ | $$x_i^{(t)} = \mu^{(t-1)} + \sigma^{(t-1)} y_i^{(t)} $$ |
| $$\mu^{(t)}$$ | mean of the generation (t) |
| $$\sigma^{(t)}$$ | step size |
| $$C^{(t)}$$ | covariance matrix |
| $$B^{(t)}$$ | a matrix of $$C$$'s eigenvectors as row vectors |
| $$D^{(t)}$$ | a diagonal matrix with $$C$$'s eigenvalues on the diagnose. |
| $$p_\sigma^{(t)}$$ | evaluation path for $$\sigma$$ at the generation (t) |
| $$p_c^{(t)}$$ | evaluation path for $$C$$ at the generation (t) |
| $$\alpha_\mu$$ | learning rate for $$\mu$$'s update |
| $$\alpha_\sigma$$ | learning rate for $$p_\sigma$$ |
| $$d_\sigma$$ | damping factor for $$\sigma$$'s update |
| $$\alpha_{cp}$$ | learning rate for $$p_c$$ |
| $$\alpha_{c\lambda}$$ | learning rate for $$C$$'s rank-min(λ, n) update |
| $$\alpha_{c1}$$ | learning rate for $$C$$'s rank-1 update |



### Updating the Mean

$$
\mu^{(t+1)} = \mu^{(t)} + \alpha_\mu \frac{1}{\lambda}\sum_{i=1}^\lambda (x_i^{(t+1)} - \mu^{(t)})
$$

CMA-ES has a learning rate $$\alpha_\mu \leq 1$$ to control how fast the mean $$\mu$$ should be updated.  Usually it is set to 1 and thus the equation becomes the same as in vanilla ES, $$\mu^{(t+1)} = \frac{1}{\lambda}\sum_{i=1}^\lambda (x_i^{(t+1)}$$.



### Controlling the Step Size

The sampling process can be decoupled from the mean and standard deviation:

$$
x^{(t+1)}_i = \mu^{(t)} + \sigma^{(t)} y^{(t+1)}_i \text{, where } y^{(t+1)}_i = \frac{x_i^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}} \sim \mathcal{N}(0, C)
$$


The parameter $$\sigma$$ controls the overall scale of the distribution. It is separated from the covariance matrix so that we can change steps faster than the full covariance. A larger step size leads to faster parameter update. In order to evaluate whether the current step size is proper, CMA-ES constructs an *evolution path* $$p_\sigma$$ by summing up a consecutive sequence of moving steps, $$\frac{1}{\lambda}\sum_{i}^\lambda y_i^{(j)}, j=1, \dots, t$$. By comparing this path length with its expected length under random selection (meaning single steps are uncorrelated), we are able to adjust $$\sigma$$ accordingly (See Fig. 2). 


![CMA-ES step size]({{ '/assets/images/CMA-ES-step-size-path.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. Three scenarios of how single steps are correlated in different ways and their impacts on step size update. (Image source: additional annotations on Fig 5 in [CMA-ES tutorial](https://arxiv.org/abs/1604.00772) paper)*

Each time the evolution path is updated with the average of moving step $$y_i$$ in the same generation.

$$
\begin{aligned}
&\frac{1}{\lambda}\sum_{i=1}^\lambda y_i^{(t+1)} 
= \frac{1}{\lambda} \frac{\sum_{i=1}^\lambda x_i^{(t+1)} - \lambda \mu^{(t)}}{\sigma^{(t)}}
= \frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}} \\
&\frac{1}{\lambda}\sum_{i=1}^\lambda y_i^{(t+1)} 
\sim \frac{1}{\lambda}\mathcal{N}(0, \lambda C^{(t)}) 
\sim \frac{1}{\sqrt{\lambda}}{C^{(t)}}^{\frac{1}{2}}\mathcal{N}(0, I) \\
&\text{Thus } \sqrt{\lambda}\;{C^{(t)}}^{-\frac{1}{2}} \frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}} \sim \mathcal{N}(0, I)
\end{aligned}
$$


> By multiplying with $$C^{-\frac{1}{2}}$$, the evolution path is transformed to be independent of its direction. The term $${C^{(t)}}^{-\frac{1}{2}} = {B^{(t)}}^\top {D^{(t)}}^{-\frac{1}{2}} {B^{(t)}}$$ transformation works as follows:
1. $${B^{(t)}}$$ contains row vectors of $$C$$'s eigenvectors. It projects the original space onto the perpendicular principal axes.
2. Then $${D^{(t)}}^{-\frac{1}{2}} = \text{diag}(\frac{1}{\lambda_1}, \dots, \frac{1}{\lambda_n})$$ scales the length of principal axes to be equal.
3. $${B^{(t)}}^\top$$ transforms the space back to the original coordinate system.

In order to assign higher weights to recent generations, we use polyak averaging to update the evolution path with learning rate $$\alpha_\sigma$$. Meanwhile, the weights are balanced so that $$p_\sigma$$ is [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior), $$\sim \mathcal{N}(0, I)$$ both before and after one update.


$$
\begin{aligned}
p_\sigma^{(t+1)} 
& = (1 - \alpha_\sigma) p_\sigma^{(t)} + \sqrt{1 - (1 - \alpha_\sigma)^2}\;\sqrt{\lambda}\; {C^{(t)}}^{-\frac{1}{2}} \frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}} \\
& = (1 - \alpha_\sigma) p_\sigma^{(t)} + \sqrt{c_\sigma (2 - \alpha_\sigma)\lambda}\;{C^{(t)}}^{-\frac{1}{2}} \frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}}
\end{aligned}
$$


The expected length of $$p_\sigma$$ under random selection is $$\mathbb{E}\|\mathcal{N}(0,I)\|$$, that is the expectation of the L2-norm of a $$\mathcal{N}(0,I)$$ random variable. Following the idea in Fig. 2, we adjust the step size according to the ratio of $$\|p_\sigma^{(t+1)}\| / \mathbb{E}\|\mathcal{N}(0,I)\|$$:


$$
\begin{aligned}
\ln\sigma^{(t+1)} &= \ln\sigma^{(t)} + \frac{\alpha_\sigma}{d_\sigma} \Big(\frac{\|p_\sigma^{(t+1)}\|}{\mathbb{E}\|\mathcal{N}(0,I)\|} - 1\Big) \\
\sigma^{(t+1)} &= \sigma^{(t)} \exp\Big(\frac{\alpha_\sigma}{d_\sigma} \Big(\frac{\|p_\sigma^{(t+1)}\|}{\mathbb{E}\|\mathcal{N}(0,I)\|} - 1\Big)\Big)
\end{aligned}
$$

where $$d_\sigma \approx 1$$ is a damping parameter, scaling how fast $$\ln\sigma$$ should be changed. 



### Adapting the Covariance Matrix 

For the covariance matrix, it can be estimated from scratch using $$y_i$$ of elite samples (recall that $$y_i \sim \mathcal{N}(0, C)$$):

$$
C_\lambda^{(t+1)} 
= \frac{1}{\lambda}\sum_{i=1}^\lambda y^{(t+1)}_i {y^{(t+1)}_i}^\top
= \frac{1}{\lambda {\sigma^{(t)}}^2} \sum_{i=1}^\lambda (x_i^{(t+1)} - \mu^{(t)})(x_i^{(t+1)} - \mu^{(t)})^\top
$$

The above estimation is only reliable when the selected population is large enough. However, we do want to run *fast* iteration with a *small* population of samples in each generation. That's why CMA-ES invented a more reliable but also more complicated way to update $$C$$. It involves two independent routes, 
- *Rank-min(λ, n) update*: uses the history of $$\{C_\lambda\}$$, each estimated from scratch in one generation.
- *Rank-one update*: estimates the moving steps $$y_i$$ and the sign information from the history.

The first route considers the estimation of $$C$$ from the entire history of $$\{C_\lambda\}$$. For example, if we have experienced a large number of generations, $$C^{(t+1)} \approx \text{avg}(C_\lambda^{(i)}; i=1,\dots,t)$$ would be a good estimator. Similar to $$p_\sigma$$, we also use polyak averaging with a learning rate to incorporate the history:

$$
C^{(t+1)} 
= (1 - \alpha_{c\lambda}) C^{(t)} + \alpha_{c\lambda} C_\lambda^{(t+1)}
= (1 - \alpha_{c\lambda}) C^{(t)} + \alpha_{c\lambda} \frac{1}{\lambda} \sum_{i=1}^\lambda y^{(t+1)}_i {y^{(t+1)}_i}^\top
$$

A common choice for the learning rate is $$\alpha_{c\lambda} \approx \min(1, \lambda/n^2)$$.

The second route tries to solve the issue that $$y_i{y_i}^\top = (-y_i)(-y_i)^\top$$ loses the sign information. Similar to how we adjust the step size $$\sigma$$, an evolution path $$p_c$$ is used to track the sign information and it is constructed in a way that $$p_c$$ is conjugate, $$\sim \mathcal{N}(0, C)$$ both before and after a new generation. 

We may consider $$p_c$$ as another way to compute $$\text{avg}_i(y_i)$$ (notice that both $$\sim \mathcal{N}(0, C)$$) while the entire history is used and the sign information is maintained. Note that we've known $$\sqrt{k}\frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}} \sim \mathcal{N}(0, C)$$ in the [last section](#controlling-the-step-size), 

$$
\begin{aligned}
p_c^{(t+1)} 
&= (1-\alpha_{cp}) p_c^{(t)} + \sqrt{1 - (1-\alpha_{cp})^2}\;\sqrt{\lambda}\;\frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}} \\
&= (1-\alpha_{cp}) p_c^{(t)} + \sqrt{\alpha_{cp}(2 - \alpha_{cp})\lambda}\;\frac{\mu^{(t+1)} - \mu^{(t)}}{\sigma^{(t)}}
\end{aligned}
$$

Then the covariance matrix is updated according to $$p_c$$:

$$
C^{(t+1)} = (1-\alpha_{c1}) C^{(t)} + \alpha_{c1}\;p_c^{(t+1)} {p_c^{(t+1)}}^\top
$$

The *rank-one update* approach is claimed to generate a significant improvement over the *rank-min(λ, n)-update* when $$k$$ is small, because the signs of moving steps and correlations between consecutive steps are all utilized and passed down through generations.

Eventually we combine two approaches together,

$$
C^{(t+1)} 
= (1 - \alpha_{c\lambda} - \alpha_{c1}) C^{(t)}
+ \alpha_{c1}\;\underbrace{p_c^{(t+1)} {p_c^{(t+1)}}^\top}_\textrm{rank-one update}
+ \alpha_{c\lambda} \underbrace{\frac{1}{\lambda} \sum_{i=1}^\lambda y^{(t+1)}_i {y^{(t+1)}_i}^\top}_\textrm{rank-min(lambda, n) update}
$$


![CMA-ES Algorithm]({{ '/assets/images/CMA-ES-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}

In all my examples above, each elite sample is considered to contribute an equal amount of weights, $$1/\lambda$$. The process can be easily extended to the case where selected samples are assigned with different weights, $$w_1, \dots, w_\lambda$$, according to their performances. See more detail in [tutorial](https://arxiv.org/abs/1604.00772).


![CMA-ES Illustration]({{ '/assets/images/CMA-ES-illustration.png' | relative_url }})
{: style="width: 80%;" class="center"}
*Fig. 3. Illustration of how CMA-ES works on a 2D optimization problem (the lighter color the better). Black dots are samples in one generation. The samples are more spread out initially but when the model has higher confidence in finding a good solution in the late stage, the samples become very concentrated over the global optimum. (Image source: [Wikipedia CMA-ES](https://en.wikipedia.org/wiki/CMA-ES))*



## Natural Evolution Strategies

Natural Evolution Strategies (**NES**; [Wierstra, et al, 2008](https://arxiv.org/abs/1106.4487)) optimizes in a search distribution of parameters and moves the distribution in the direction of high fitness indicated by the *natural gradient*.


### Natural Gradients

Given an objective function $$\mathcal{J}(\theta)$$ parameterized by $$\theta$$, let's say our goal is to find the optimal $$\theta$$ to maximize the objective function value. A *plain gradient* finds the steepest direction within a small Euclidean distance from the current $$\theta$$; the distance restriction is applied on the parameter space. In other words, we compute the plain gradient with respect to a small change of the absolute value of $$\theta$$. The optimal step is:


$$
d^{*} = \operatorname*{argmax}_{\|d\| = \epsilon} \mathcal{J}(\theta + d)\text{, where }\epsilon \to 0
$$

Differently, *natural gradient* works with a probability [distribution](https://arxiv.org/abs/1301.3584v7) [space](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) parameterized by $$\theta$$, $$p_\theta(x)$$ (referred to as "search distribution" in NES [paper](https://arxiv.org/abs/1106.4487)). It looks for the steepest direction within a small step in the distribution space where the distance is measured by KL divergence. With this constraint we ensure that each update is moving along the distributional manifold with constant speed, without being slowed down by its curvature.


$$
d^{*}_\text{N} = \operatorname*{argmax}_{\text{KL}[p_\theta \| p_{\theta+d}] = \epsilon} \mathcal{J}(\theta + d)
$$



### Estimation using Fisher Information Matrix

But, how to compute $$\text{KL}[p_\theta \| p_{\theta+\Delta\theta}]$$ precisely? By running Taylor expansion of $$\log p_{\theta + d}$$ at $$\theta$$, we get:


$$
\begin{aligned}
& \text{KL}[p_\theta \| p_{\theta+d}] \\
&= \mathbb{E}_{x \sim p_\theta} [\log p_\theta(x) - \log p_{\theta+d}(x)] & \\
&\approx \mathbb{E}_{x \sim p_\theta} [ \log p_\theta(x) -( \log p_{\theta}(x) + \nabla_\theta \log p_{\theta}(x) d + \frac{1}{2}d^\top \nabla^2_\theta \log p_{\theta}(x) d)] & \scriptstyle{\text{; Taylor expand }\log p_{\theta+d}} \\
&\approx - \mathbb{E}_x [\nabla_\theta \log p_{\theta}(x)] d - \frac{1}{2}d^\top \mathbb{E}_x [\nabla^2_\theta \log p_{\theta}(x)] d & 
\end{aligned}
$$

where

$$
\begin{aligned}
\mathbb{E}_x [\nabla_\theta \log p_{\theta}] d 
&= \int_{x\sim p_\theta} p_\theta(x) \nabla_\theta \log p_\theta(x) & \\
&= \int_{x\sim p_\theta} p_\theta(x) \frac{1}{p_\theta(x)} \nabla_\theta p_\theta(x) & \\
&= \nabla_\theta \Big( \int_{x} p_\theta(x) \Big) & \scriptstyle{\textrm{; note that }p_\theta(x)\textrm{ is probability distribution.}} \\
&= \nabla_\theta (1) = 0
\end{aligned}
$$

Finally we have,

$$
\text{KL}[p_\theta \| p_{\theta+d}] = - \frac{1}{2}d^\top \mathbf{F}_\theta d 
\text{, where }\mathbf{F}_\theta = \mathbb{E}_x [(\nabla_\theta \log p_{\theta}) (\nabla_\theta \log p_{\theta})^\top]
$$

where $$\mathbf{F}_\theta$$ is called the **[Fisher Information Matrix](http://mathworld.wolfram.com/FisherInformationMatrix.html)** and [it is](https://wiseodd.github.io/techblog/2018/03/11/fisher-information/) the covariance matrix of $$\nabla_\theta \log p_\theta$$ since $$\mathbb{E}[\nabla_\theta \log p_\theta] = 0$$. 

The solution to the following optimization problem:

$$
\max \mathcal{J}(\theta + d) \approx \max \big( \mathcal{J}(\theta) + {\nabla_\theta\mathcal{J}(\theta)}^\top d \big)\;\text{ s.t. }\text{KL}[p_\theta \| p_{\theta+d}] - \epsilon = 0
$$

can be found using a Lagrangian multiplier,

$$
\begin{aligned}
\mathcal{L}(\theta, d, \beta) &= \mathcal{J}(\theta) + \nabla_\theta\mathcal{J}(\theta)^\top d - \beta (\frac{1}{2}d^\top \mathbf{F}_\theta d + \epsilon) = 0 \text{ s.t. } \beta > 0 \\
\nabla_d \mathcal{L}(\theta, d, \beta) &= \nabla_\theta\mathcal{J}(\theta) - \beta\mathbf{F}_\theta d = 0 \\
\text{Thus } d_\text{N}^* &= \nabla_\theta^\text{N} \mathcal{J}(\theta) = \mathbf{F}_\theta^{-1} \nabla_\theta\mathcal{J}(\theta) 
\end{aligned}
$$

where $$d_\text{N}^*$$ only extracts the direction of the optimal moving step on $$\theta$$, ignoring the scalar $$\beta^{-1}$$.


![Plain vs natural coordinates]({{ '/assets/images/CMA-ES-coordinates.png' | relative_url }})
{: style="width: 90%;" class="center"}
*Fig. 4. The natural gradient samples (black solid arrows) in the right are the plain gradient samples (black solid arrows)  in the left multiplied by the inverse of their covariance. In this way, a gradient direction with high uncertainty (indicated by high covariance with other samples) are penalized with a small weight. The aggregated natural gradient (red dash arrow) is therefore more trustworthy than the natural gradient (green solid arrow). (Image source: additional annotations on Fig 2 in [NES](https://arxiv.org/abs/1106.4487) paper)*



### NES Algorithm

The fitness associated with one sample is labeled as $$f(x)$$ and the search distribution over $$x$$ is parameterized by $$\theta$$. NES is expected to optimize the parameter $$\theta$$ to achieve maximum expected fitness: 

$$
\mathcal{J}(\theta) = \mathbb{E}_{x\sim p_\theta(x)} [f(x)] = \int_x f(x) p_\theta(x) dx
$$

Using the same log-likelihood [trick](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/) in [REINFORCE]({{ site.baseurl }}{% post_url 2018-04-08-policy-gradient-algorithms %}#reinforce):

$$
\begin{aligned}
\nabla_\theta\mathcal{J}(\theta) 
&= \nabla_\theta \int_x f(x) p_\theta(x) dx \\
&= \int_x f(x) \frac{p_\theta(x)}{p_\theta(x)}\nabla_\theta p_\theta(x) dx \\
& = \int_x f(x) p_\theta(x) \nabla_\theta \log p_\theta(x) dx \\
& = \mathbb{E}_{x \sim p_\theta} [f(x) \nabla_\theta \log p_\theta(x)]
\end{aligned}
$$      


![NES]({{ '/assets/images/NES-algorithm.png' | relative_url }})
{: style="width: 80%;" class="center"}


Besides natural gradients, NES adopts a couple of important heuristics to make the algorithm performance more robust.
- <a name="fitness-shaping"></a>NES applies **rank-based fitness shaping**, that is to use the *rank* under monotonically increasing fitness values instead of using $$f(x)$$ directly. Or it can be a function of the rank (“utility function”), which is considered as a free parameter of NES.
- NES adopts **adaptation sampling** to adjust hyperparameters at run time. When changing $$\theta \to \theta’$$, samples drawn from $$p_\theta$$ are compared with samples from $$p_{\theta’}$$ using [Mann-Whitney U-test(https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)]; if there shows a positive or negative sign, the target hyperparameter decreases or increases by a multiplication constant. Note the score of a sample $$x’_i \sim p_{\theta’}(x)$$ has importance sampling weights applied $$w_i’ = p_\theta(x) / p_{\theta’}(x)$$.





## Applications: ES in Deep Reinforcement Learning


### OpenAI ES for RL

The concept of using evolutionary algorithms in reinforcement learning can be traced back [long ago](https://arxiv.org/abs/1106.0221), but only constrained to tabular RL due to computational limitations.

Inspired by [NES](#natural-evolution-strategies), researchers at OpenAI ([Salimans, et al. 2017](https://arxiv.org/abs/1703.03864)) proposed to use NES as a gradient-free black-box optimizer to find optimal policy parameters $$\theta$$ that maximizes the return function $$F(\theta)$$. The key is to add Gaussian noise $\epsilon$ on the model parameter $\theta$ and then use the log-likelihood trick to write it as the gradient of the Gaussian pdf. Eventually only the noise term is left as a weighting scalar for measured performance. 

Let’s say the current parameter value is $$\hat{\theta}$$ (the added hat is to distinguish the value from the random variable $$\theta$$). The search distribution of $$\theta$$ is designed to be an isotropic multivariate Gaussian with a mean $$\hat{\theta}$$ and a fixed covariance matrix $$\sigma^2 I$$,


$$
\theta \sim \mathcal{N}(\hat{\theta}, \sigma^2 I) \text{ equivalent to } \theta = \hat{\theta} + \sigma\epsilon, \epsilon \sim \mathcal{N}(0, I)
$$

The gradient for $$\theta$$ update is:

$$
\begin{aligned}
& \nabla_\theta \mathbb{E}_{\theta\sim\mathcal{N}(\hat{\theta}, \sigma^2 I)} F(\theta) \\
&= \nabla_\theta \mathbb{E}_{\epsilon\sim\mathcal{N}(0, I)} F(\hat{\theta} + \sigma\epsilon) \\
&= \nabla_\theta \int_{\epsilon} p(\epsilon) F(\hat{\theta} + \sigma\epsilon) d\epsilon & \scriptstyle{\text{; Gaussian }p(\epsilon)=(2\pi)^{-\frac{n}{2}} \exp(-\frac{1}{2}\epsilon^\top\epsilon)} \\
&= \int_{\epsilon} p(\epsilon) \nabla_\epsilon \log p(\epsilon) \nabla_\theta \epsilon\;F(\hat{\theta} + \sigma\epsilon) d\epsilon & \scriptstyle{\text{; log-likelihood trick}}\\
&= \mathbb{E}_{\epsilon\sim\mathcal{N}(0, I)} [ \nabla_\epsilon \big(-\frac{1}{2}\epsilon^\top\epsilon\big) \nabla_\theta \big(\frac{\theta - \hat{\theta}}{\sigma}\big) F(\hat{\theta} + \sigma\epsilon) ] & \\
&= \mathbb{E}_{\epsilon\sim\mathcal{N}(0, I)} [ (-\epsilon) (\frac{1}{\sigma}) F(\hat{\theta} + \sigma\epsilon) ] & \\
&= \frac{1}{\sigma}\mathbb{E}_{\epsilon\sim\mathcal{N}(0, I)} [ \epsilon F(\hat{\theta} + \sigma\epsilon) ] & \scriptstyle{\text{; negative sign can be absorbed.}}
\end{aligned}
$$

In one generation, we can sample many $$epsilon_i, i=1,\dots,n$$ and evaluate the fitness *in parallel*. One beautiful design is that no large model parameter needs to be shared. By only communicating the random seeds between workers, it is enough for the master node to do parameter update. This approach is later extended to adaptively learn a loss function; see my previous post on [Evolved Policy Gradient]({{ site.baseurl }}{% post_url 2019-06-23-meta-reinforcement-learning %}#meta-learning-the-loss-function). 


![ES for RL]({{ '/assets/images/OpenAI-ES-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 5. The algorithm for training a RL policy using evolution strategies. (Image source: [ES-for-RL](https://arxiv.org/abs/1703.03864) paper)*

To make the performance more robust, OpenAI ES adopts virtual batch normalization (BN with mini-batch used for calculating statistics fixed), mirror sampling (sampling a pair of $$(-\epsilon, \epsilon)$$ for evaluation), and [fitness shaping](#fitness-shaping).







### Exploration with ES 

Exploration ([vs exploitation]({{ site.baseurl }}{% post_url 2018-01-23-the-multi-armed-bandit-problem-and-its-solutions %}#exploitation-vs-exploration)) is an important topic in RL. The optimization direction in the ES algorithm [above](TBA) is only extracted from the cumulative return $$F(\theta)$$. Without explicit exploration, the agent might get trapped in a local optimum.




Novelty-Search ES (**NS-ES**; [Conti et al, 2018](https://arxiv.org/abs/1712.06560)) encourages exploration by updating the parameter in the direction to maximize the *novelty* score. The novelty score depends on a domain-specific behavior characterization function $$b(\pi_\theta)$$. The choice of $$b(\pi_\theta)$$ is specific to the task and seems to be a bit arbitrary; for example, in the Humanoid locomotion task in the paper, $$b(\pi_\theta)$$ is the final $$(x,y)$$ location of the agent.
1. Every policy's $$b(\pi_\theta)$$ is pushed to an archive set $$\mathcal{A}$$.
2. Novelty of a policy $$\pi_\theta$$ is measured as the k-nearest neighbor score between $$b(\pi_\theta)$$ and all other entries in $$\mathcal{A}$$.
(The use case of the archive set sounds quite similar to [episodic memory]({{ site.baseurl }}{% post_url 2019-06-23-meta-reinforcement-learning %}#episodic-control).)


$$
N(\theta, \mathcal{A}) = \frac{1}{\lambda} \sum_{i=1}^\lambda \| b(\pi_\theta), b^\text{knn}_i \|_2
\text{, where }b^\text{knn}_i \in \text{kNN}(b(\pi_\theta), \mathcal{A})
$$

The ES optimization step relies on the novelty score instead of fitness:

$$
\nabla_\theta \mathbb{E}_{\theta\sim\mathcal{N}(\hat{\theta}, \sigma^2 I)} N(\theta, \mathcal{A})
= \frac{1}{\sigma}\mathbb{E}_{\epsilon\sim\mathcal{N}(0, I)} [ \epsilon N(\hat{\theta} + \sigma\epsilon, \mathcal{A}) ]
$$

NS-ES maintains a group of $$M$$ independently trained agents ("meta-population"), $$\mathcal{M} = \{\theta_1, \dots, \theta_M \}$$ and picks one to advance proportional to the novelty score. Eventually we select the best policy. This process is equivalent to ensembling; also see the same idea in [SVPG]({{ site.baseurl }}{% post_url 2018-04-08-policy-gradient-algorithms %}#svpg).

$$
\begin{aligned}
m &\leftarrow \text{pick } i=1,\dots,M\text{ according to probability}\frac{N(\theta_i, \mathcal{A})}{\sum_{j=1}^M N(\theta_j, \mathcal{A})} \\
\theta_m^{(t+1)} &\leftarrow \theta_m^{(t)} + \alpha \frac{1}{\sigma}\sum_{i=1}^N \epsilon_i N(\theta^{(t)}_m + \epsilon_i, \mathcal{A}) \text{ where }\epsilon_i \sim \mathcal{N}(0, I)
\end{aligned}
$$

where $$N$$ is the number of Gaussian perturbation noise vectors and $$\alpha$$ is the learning rate.

NS-ES completely discards the reward function and only optimizes for novelty to avoid deceptive local optima. To incorporate the fitness back into the formula, another two variations are proposed.

**NSR-ES**:

$$
\theta_m^{(t+1)} \leftarrow \theta_m^{(t)} + \alpha \frac{1}{\sigma}\sum_{i=1}^N \epsilon_i \frac{N(\theta^{(t)}_m + \epsilon_i, \mathcal{A}) + F(\theta^{(t)}_m + \epsilon_i)}{2}
$$


**NSRAdapt-ES (NSRA-ES)**: the adaptive weighting parameter $$w = 1.0$$ initially. We start decreasing $$w$$ if performance stays flat for a number of generations. Then when the performance starts to increase, we stop decreasing $$w$$ but increase it instead. In this way, fitness is preferred when the performance stops growing but novelty is preferred otherwise. 

$$
\theta_m^{(t+1)} \leftarrow \theta_m^{(t)} + \alpha \frac{1}{\sigma}\sum_{i=1}^N \epsilon_i \big((1-w) N(\theta^{(t)}_m + \epsilon_i, \mathcal{A}) + w F(\theta^{(t)}_m + \epsilon_i)\big)
$$


![NS-ES Experiments]({{ '/assets/images/NS-ES-experiments.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 6. (Left) The environment is Humanoid locomotion with a three-sided wall which plays a role as a deceptive trap to create local optimum. (Right) Experiments compare ES baseline and other variations that encourage exploration. (Image source: [NS-ES](https://arxiv.org/abs/1712.06560) paper)*



### CEM-RL

![CEM-RL]({{ '/assets/images/CEM-RL.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Architectures of the (a) CEM-RL and (b) [ERL](https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf) algorithms (Image source: [CEM-RL](https://arxiv.org/abs/1810.01222) paper)* 


The CEM-RL method ([Pourchot & Sigaud, 2019](https://arxiv.org/abs/1810.01222)) combines Cross Entropy Method (CEM) with either [DDPG]({{ site.baseurl }}{% post_url 2018-04-08-policy-gradient-algorithms %}#ddpg) or [TD3]({{ site.baseurl }}{% post_url 2018-04-08-policy-gradient-algorithms %}#td3). CEM here works pretty much the same as the simple Gaussian ES described [above](#simple-gaussian-evolution-strategies) and therefore the same function can be replaced using CMA-ES. CEM-RL is built on the framework of *Evolutionary Reinforcement Learning* (*ERL*; [Khadka & Tumer, 2018](https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf)) in which the standard EA algorithm selects and evolves a population of actors and the rollout experience generated in the process is then added into reply buffer for training both RL-actor and RL-critic networks.

Workflow:
- 1) The mean actor of the CEM population is $$\pi_\mu$$ is initialized with a random actor network.
- 2) The critic network $$Q$$ is initialized too, which will be updated by DDPG/TD3.
- 3) Repeat until happy:
    - a. Sample a population of actors $$\sim \mathcal{N}(\pi_\mu, \Sigma)$$.
    - b. Half of the population is evaluated. Their fitness scores are used as the cumulative reward $$R$$ and added into replay buffer.
    - c. The other half are updated together with the critic.
    - d. The new $$\pi_mu$$ and $$\Sigma$$ is computed using top performing elite samples. [CMA-ES](#covariance-matrix-adaptation-evolution-strategies-cma-es) can be used for parameter update too.



## Extension: EA in Deep Learning

(This section is not on evolution strategies, but still an interesting and relevant reading.)


The *Evolutionary Algorithms* have been applied on many deep learning problems. POET ([Wang et al, 2019](https://arxiv.org/abs/1901.01753)) is a framework based on EA and attempts to generate a variety of different tasks while the problems themselves are being solved. POET has been introduced in my [last post]({{ site.baseurl }}{% post_url 2019-06-23-meta-reinforcement-learning %}#task-generation-by-domain-randomization) on meta-RL. Evolutionary Reinforcement Learning (ERL) is another example; See Fig. 7 (b).

Below I would like to introduce two applications in more detail, *Population-Based Training (PBT)* and *Weight-Agnostic Neural Networks (WANN)*.


### Hyperparameter Tuning: PBT

![PBT]({{ '/assets/images/PBT.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 8. Paradigms of comparing different ways of hyperparameter tuning. (Image source: [PBT](https://arxiv.org/abs/1711.09846) paper)*

Population-Based Training ([Jaderberg, et al, 2017](https://arxiv.org/abs/1711.09846)), short for **PBT** applies EA on the problem of hyperparameter tuning. It jointly trains a population of models and corresponding hyperparameters for optimal performance.  

PBT starts with a set of random candidates, each containing a pair of model weights initialization and hyperparameters, $$\{(\theta_i, h_i)\mid i=1, \dots, N\}$$. Every sample is trained in parallel and asynchronously evaluates its own performance periodically. Whenever a member deems ready (i.e. after taking enough gradient update steps, or when the performance is good enough), it has a chance to be updated by comparing with the whole population:
- **`exploit()`**: When this model is under-performing, the weights could be replaced with a better performing model.
- **`explore()`**: If the model weights are overwritten, `explore` step perturbs the hyperparameters with random noise.

In this process, only promising model and hyperparameter pairs can survive and keep on evolving, achieving better utilization of computational resources. 


![PBT Algorithm]({{ '/assets/images/PBT-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 9. The algorithm of population-based training. (Image source: [PBT](https://arxiv.org/abs/1711.09846) paper)* 


### Network Topology Optimization: WANN

*Weight Agnostic Neural* Networks (short for **WANN**; [Gaier & Ha 2019](https://arxiv.org/abs/1906.04358)) experiments with searching for the smallest network topologies that can achieve the optimal performance without training the network weights. By not considering the best configuration of network weights, WANN puts much more emphasis on the architecture itself, making the focus different from [NAS](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf). WANN is heavily inspired by a classic genetic algorithm to evolve network topologies, called *NEAT* ("Neuroevolution of Augmenting Topologies"; [Stanley & Miikkulainen 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.gecco02_1.pdf)).

The workflow of WANN looks pretty much the same as standard GA:
1. Initialize: Create a population of minimal networks.
2. Evaluation: Test with a range of *shared* weight values.
3. Rank and Selection: Rank by performance and complexity.
4. Mutation: Create new population by varying best networks.


![Mutation operations in WANN]({{ '/assets/images/WANN-mutations.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 10. mutation operations for searching for new network topologies in WANN (Image source: [WANN](https://arxiv.org/abs/1906.04358) paper)*


At the "evaluation" stage, all the network weights are set to be the same. In this way, WANN is actually searching for network that can be described with a minimal description length. In the "selection" stage, both the network connection and the model performance are considered.


![WANN results]({{ '/assets/images/WANN-results.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 11. Performance of WANN found network topologies on different RL tasks are compared with baseline FF networks commonly used in the literature. "Tuned Shared Weight" only requires adjusting one weight value. (Image source: [WANN](https://arxiv.org/abs/1906.04358) paper)*

As shown in Fig. 11, WANN results are evaluated with both random weights and shared weights (single weight). It is interesting that even when enforcing weight-sharing on all weights and tuning this single parameter, WANN can discover topologies that achieve non-trivial good performance.


---
Cited as:
```
@article{weng2019ES,
  title   = "Evolution Strategies",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io/lil-log",
  year    = "2019",
  url     = "https://lilianweng.github.io/lil-log/2019/09/05/evolution-strategies.html"
}
```

## References

[1] Nikolaus Hansen. ["The CMA Evolution Strategy: A Tutorial"](https://arxiv.org/abs/1604.00772) arXiv preprint arXiv:1604.00772 (2016).

[2] Marc Toussaint. [Slides: "Introduction to Optimization"](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/teaching/13-Optimization/06-blackBoxOpt.pdf)

[3] David Ha. ["A Visual Guide to Evolution Strategies"](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) blog.otoro.net. Oct 2017.

[4] Daan Wierstra, et al. ["Natural evolution strategies."](https://arxiv.org/abs/1106.4487) IEEE World Congress on Computational Intelligence, 2008.

[5] Agustinus Kristiadi. ["Natural Gradient Descent"](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) Mar 2018.

[6] Razvan Pascanu & Yoshua Bengio. ["Revisiting Natural Gradient for Deep Networks."](https://arxiv.org/abs/1301.3584v7) arXiv preprint arXiv:1301.3584 (2013).

[7] Tim Salimans, et al. ["Evolution strategies as a scalable alternative to reinforcement learning."](https://arxiv.org/abs/1703.03864) arXiv preprint arXiv:1703.03864 (2017).

[8] Edoardo Conti, et al. ["Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents."](https://arxiv.org/abs/1712.06560) NIPS. 2018.

[9] Aloïs Pourchot & Olivier Sigaud. ["CEM-RL: Combining evolutionary and gradient-based methods for policy search."](https://arxiv.org/abs/1810.01222) ICLR 2019.

[10] Shauharda Khadka & Kagan Tumer. ["Evolution-guided policy gradient in reinforcement learning."](https://papers.nips.cc/paper/7395-evolution-guided-policy-gradient-in-reinforcement-learning.pdf) NIPS 2018.

[11] Max Jaderberg, et al. ["Population based training of neural networks."](https://arxiv.org/abs/1711.09846) arXiv preprint arXiv:1711.09846 (2017).

[12] Adam Gaier & David Ha. ["Weight Agnostic Neural Networks."](https://arxiv.org/abs/1906.04358) arXiv preprint arXiv:1906.04358 (2019).


