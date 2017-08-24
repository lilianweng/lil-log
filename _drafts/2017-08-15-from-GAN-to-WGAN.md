---
layout: post
comments: false
title: "From GAN to WGAN"
date: 2017-08-15 00:23:00
---

> TBA

<!--more-->

[Generative adversarial network](https://arxiv.org/pdf/1406.2661.pdf) has shown great results in many generative tasks to replicate the real-world examples.


{: class="table-of-content"}
* TOC
{:toc}


## Kullback–Leibler and Jensen–Shannon Divergence

Before we start looking into GAN, let us first review two metrics for quantifying the similiarity between two probability distributions.

1) [KL (Kullback–Leibler) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measures how one probability distribution $$p$$ diverges from a second expected probability distribution $$q$$.

$$
D_{KL}(p||q) = \int_x p(x) \log \frac{p(x)}{q(x)} dx
$$

$$D_{KL}$$ achieves the minimum 0 when $$p(x) == q(x)$$ everywhere.


It is noticable according to the formula that KL divergence is not symmetric. In cases where $$p(x)$$ is close to zero, but $$q(x)$$ is significantly non-zero, the $$q$$'s effect is disregarded. It could seem like a bug when we just want to measure the similarity between two equally important distributions.

2) [Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) is another measure of similarity between two probability distributions, bounded by $$[0, 1]$$. JS divergence is symmetric! and more smoothed.

https://www.quora.com/Why-isnt-the-Jensen-Shannon-divergence-used-more-often-than-the-Kullback-Leibler-since-JS-is-symmetric-thus-possibly-a-better-indicator-of-distance

$$
D_{JS}(p||q) = \frac{1}{2} D_{KL}(p || \frac{p + q}{2}) + \frac{1}{2} D_{KL}(q || \frac{p + q}{2})
$$

![KL and JS divergence]({{ '/assets/images/KL_JS_divergence.png' | relative_url }})
{: style="width: 640px;" class="center"}
*Fig. 2. KL and JS divergence. $$m=(p+q)/2$$*

One reason behind GAN's big success is switching the loss measure from assymetric KL divergence in traditional maximum-likelihood approach to symmetric JS divergence.



## Generative Adversarial Network (GAN)

GAN consists of two models:
- A discriminator $$D$$ estimates the probability of a given sample coming from the real data set. It works as a critic and is optimized to tell the fake samples from the real ones.
- A generator $$G$$ outputs synthetic samples given a noise variable input $$z$$. It is trained to captured the real data distribution so that its output samples can be as real as possible, or in other words, can trick the discriminator to produce a high probability.
 


![Generative adversarial network]({{ '/assets/images/GAN.png' | relative_url }})
{: style="width: 600px;" class="center"}
*Fig. 1. Architecture of a generative adversarial network. (Image source: http://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html)*


These two models compete against each other during training: the generator $$G$$ is trying hard to trick the discriminator while the judge model is trying hard not to be cheated. This interesting zero-sum game between these two models motivates both to improve their functionalities.

Let’s say, 

| Symbol        | Meaning           | Notes  |
| ------------- | ------------- | ------------- |
| $$p_{r}$$| Data distribution over real sample $$x$$ | |
| $$p_{z}$$ | Data distribution over noise input $$z$$ | Usually, just uniform. |
| $$p_{g}$$ | The generator’s distribution over data $$x$$ | |



On one hand, we want to make sure the discriminator $$D$$’s decisions over real data are accurate by maximizing $$\mathbb{E}_{x \sim p_{r}(x)} [\log D(x)]$$. Meanwhile, given a fake sample $$G(z), z \sim p_z(z)$$, the discriminator is expected to output a probability, $$D(G(z))$$ close to 0 by maximizing $$\mathbb{E}_{z \sim p_{z}(z)} [\log (1 - D(G(z)))]$$.

On the other hand, the generator is trained to increase the chances of $$D$$ producing a high probability for a fake example, thus to minimize $$\mathbb{E}_{z \sim p_{z}(z)} [\log (1 - D(G(z)))]$$.

When combining these two aspects together, $$D$$ and $$G$$ are playing a minimax game in which we should optimize the following loss function:

$$
\begin{aligned}
\min_G \max_D L(D, G) 
& = \mathbb{E}_{x \sim p_{r}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))] \\
& = \mathbb{E}_{x \sim p_{r}(x)} [\log D(x)] + \mathbb{E}_{x \sim p_g(x)} [\log(1 - D(x)]
\end{aligned}
$$


### What is the optimal value for $$D$$?

Now we have a well-defined loss function. Let’s first examine what is the best value for $$D$$.

$$
L(G, D) = \int_x \bigg( p_{r}(x) \log(D(x)) + p_g (x) \log(1 - D(x)) \bigg) dx
$$

Since we are interested in what is the best value of $$D(x)$$ to maximize $$L(G, D)$$, let us label 

$$
\tilde{x} = D(x), 
A=p_{r}(x), 
B=p_g(x)
$$

And then what is inside the integral (we can safely ignore the integral because $$x$$ is sampled over all the possible values) is:

$$
\begin{aligned}
f(\tilde{x}) 
& = A log\tilde{x} + B log(1-\tilde{x}) \\

\frac{d f(\tilde{x})}{d \tilde{x}}
& = A \frac{1}{ln10} \frac{1}{\tilde{x}} - B \frac{1}{ln10} \frac{1}{1 - \tilde{x}} \\
& = \frac{1}{ln10} (\frac{A}{\tilde{x}} - \frac{B}{1-\tilde{x}}) \\
& = \frac{1}{ln10} \frac{A - (A + B)\tilde{x}}{\tilde{x} (1 - \tilde{x})} \\
\end{aligned}
$$

Thus, set $$\frac{d f(\tilde{x})}{d \tilde{x}} = 0$$, we get the best value of the discriminator: $$D^*(x) = \tilde{x}^* = \frac{A}{A + B} = \frac{p_{r}(x)}{p_{r}(x) + p_g(x)} \in [0, 1]$$.

Once the generator is trained to its optimal, $$p_g$$ gets very close to $$p_{r}$$, $$p_g = p_{r}$$, and thus $$D^*(x)$$ becomes $$1/2$$.


### What is the global optimal?

When both $$G$$ and $$D$$ are at their optimal values, we have $$p_g = p_{r}$$ and $$D^*(x) = 1/2$$ and the value function

$$
\begin{aligned}
L(G, D^*) 
&= \int_x \bigg( p_{r}(x) \log(D^*(x)) + p_g (x) \log(1 - D^*(x)) \bigg) dx \\
&= \int_x (p_{r}(x) \log \frac{p_{r}(x)}{p_{r}(x) + p_g(x)} + p_g (x) \log \frac{p_g(x)}{p_{r}(x) + p_g(x)}) dx \\
&= \log \frac{1}{2} \int_x p_{r}(x) dx + \log \frac{1}{2} \int_x p_g(x) dx \\
&= -\log4
\end{aligned}
$$



### What does the value function represent?

The JS divergence between $$p_{r}$$ and $$p_g$$ can be computed as:

$$
\begin{aligned}
D_{JS}(p_{r} \vert\vert p_g) 
=& \frac{1}{2} D_{KL}(p_{r} || \frac{p_{r} + p_g}{2}) + \frac{1}{2} D_{KL}(p_{r} || \frac{p_{r} + p_g}{2}) \\
=& \frac{1}{2} \bigg( \log2 + \int_x p_{r}(x) \log \frac{p_{r}(x)}{p_{r} + p_g(x)} dx \bigg) + \\& \frac{1}{2} \bigg( \log2 + \int_x p_g(x) \log \frac{p_g(x)}{p_{r} + p_g(x)} dx \bigg) \\
=& \frac{1}{2} \bigg( \log4 + V(G, D^*) \bigg)
\end{aligned}
$$

Thus, 

$$
L(G, D^*) = 2D_{JS}(p_{r} \vert\vert p_g) - 2\log2
$$. 

Essentially the loss function of GAN quantifies the similarity between the generator data distribution $$p_g$$ and the real sample distribution $$p_{r}$$ by JS divergence when the discriminator is optimal. The best $$G$$ that replicates the real data distribution leads to the minimum $$L(G, D^*) = -log4$$ which is aligned with eq(x).



**Other Variations of GAN**

There are many variations of GAN in different contexts or designed for different tasks. For example, for semi-supervised learning, one idea is to update the $$D$$ outputs real class labels $$1, \dots, K$$, and one fake class label $$K$$.


**Implementation**: A very popular implementation: [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)




## Problems in GAN

Although GAN has shown great success in the realistic image generation, it is very hard to train a GAN: the training process is slow and very unstable.


### Hard to get Nash equilibrium

Salimans et al. (2016) discussed the problem with GAN's gradient-descent-based training. Two models in GAN are trained to find a [Nash equilibrium](https://en.wikipedia.org/wiki/Nash_equilibrium) to a two-player non-cooperative game. However, each model updates its cost independently with no respect to another player in the game. Updating the gradient of both models simultaneously cannot guarantee a convergence.

Let's check a simple example. While one player is controlling $$x$$ to minimize $$f_1(x) = xy$$, the other player is consistently changing $$y$$ to minimize $$f_2(y) = -xy$$.

Because $$\frac{\partial f_1}{\partial x} = y$$ and $$\frac{\partial f_2}{\partial y} = -x$$, we update $$x$$ with $$x-r \cdot y$$ and $$y$$ with $$y+ r \cdot x$$ simultanously in one iteration, where $$r$$ is the learning rate. Once $$x$$ and $$y$$ have different signs, each update would cause huge oscillation, as shown in Fig. x. 

![Nash equilibrium example]({{ '/assets/images/nash_equilibrium.png' | relative_url }})
{: style="width: 680px;" class="center"}
*Fig. x. A simulation of our example for updating $$x$$ to minimize $$xy$$ and updating $$y$$ to minimize $$-xy$$. The learning rate $$r = 0.1$$. With more iterations, the oscillation grows more and more unstable.*


### Low Dimensional Data

| Term         | Explanation  |
| ------------ | ------------ |
| [Manifold](https://en.wikipedia.org/wiki/Manifold) | A topological space that locally resembles Euclidean space near each point. Precisely, when this Euclidean space is of **dimension $$n$$**, the manifold is referred as **$$n$$-manifold**. |
| [Support](https://en.wikipedia.org/wiki/Support_(mathematics)) | A real-valued function $$f$$ is the subset of the domain containing those elements which are not mapped to **zero**.

Arjovsky and Bottou discussed the issue of the supports of $$p_r$$ and $$p_g$$ lie on low dimensional manifolds and how it contributes to the instability in GAN training thoroughly in a pretty theoretical paper ["Towards principled methods for training generative adversarial networks"](https://arxiv.org/pdf/1701.04862.pdf).

$$p_r$$ is expected to concentrate on a low dimensionl dimension, because the dimensions of many real-world datasets only appear to be aritificially high and they can well be represented in a lower dimensional space. This is actually the foundamental assumption for [Manifold Learning](http://scikit-learn.org/stable/modules/manifold.html). 
Think of the cases of the real world images. Once the theme or the contained object is fixed, the images have a lot of restrictions to follow, i.e., a dogs should have two ears and a tail, a skyscrapper should have a straight and tall body, etc. These restrictions keep images aways from the possibility of having a high-dimensional free form distribution.

$$p_g$$ lies in a low dimensional manifolds, too. Whenever the generator is asked to a much larger image like 64x64 given a small dimension, such as 100, noise variable input $$z$$, the distribution of colors over these 4096 pixels has been defined by the small 100-dimension random number vector and can hardly fill up the whole high dimensional space.


Because both $$p_g$$ and $$p_{r}$$ lie in low dimensional manifolds, they are almost certainly gonna be disjoint (See Fig. x). When they have disjoint supports, we are capable of finding a perfect discriminator that separates real and fake samples 100% correctly. Check the [paper](https://arxiv.org/pdf/1701.04862.pdf) if you are curious about the proof.


![Low dimension manifolds in high dimension space]({{ '/assets/images/low_dim_manifold.png' | relative_url }})
{: style="width: 600px;" class="center"}
*Fig. x. Low dimension manifolds in high dimension space. (Left) Two lines in a three-dimension space. (Right) Two surfaces in a three-dimension space.$$*


### Vanishing Gradient

When the discriminator is perfect, we are guaranteed with $$D(x) = 1, \forall x \in p_r$$ and $$D(x) = 1, \forall x \in p_g$$. Therefore the loss function $$L$$ falls to zero and we have no gradient to update the loss during learning iterations. Fig. x demonstrates an experiment when the discriminator gets better, the gradient vanishes fast.

![Low dimension manifolds in high dimension space]({{ '/assets/images/GAN_vanishing_gradient.png' | relative_url }})
{: style="width: 480px;" class="center"}
*Fig. x. First, a DCGAN is trained for 1, 10 and 25 epochs. Then, with the **generator fixed**, a discriminator is trained from scratch and measure the gradients with the original cost function. We see the gradient norms **decay quickly** (in log scale), in the best case 5 orders of magnitude after 4000 discriminator iterations. (Image source: [Arjovsky and Bottou, 2017](https://arxiv.org/pdf/1701.04862.pdf))*


As a result, training a GAN faces a dilemma:
- If the discriminator $$D$$ behaves badly, the generator does not have accurate feedback and the loss function cannot represent the reality.
- If the discriminator $$D$$ does a great job, the gradient of the loss function drops down to close to zero and the learning becomes super slow.

This dilemma clearly can make the GAN training very difficult.


### Model Collapse

xxx

### Lack of a proper evaluation metric

Evaluation metric: Inception score



## Improved GAN Training

First five methods are practical techniques to achieve faster convergent of GAN training, proposed in [Salimans et al. 2016](http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf).
The last two are proposed ([Arjovsky and Bottou, 2017](https://arxiv.org/pdf/1701.04862.pdf)) to solve the problem of disjoint distributions.

1. **Feature matching** xxx

2. **Minibatch discrimination**

3. Historical averaging

4. One-sided label smoothing

5. Virtual batch normalization

6. **Adding noises**. Based on the discussion in preivous section, we now know $$p_r$$ and $$p_g$$ are disjoint in a high dimensional space and it causes the vanishing gradient issue. To artificially smoothen the distribution and to create higher chances for two probability distributions to have overlaps, one solution proposed is to add continous noises onto the inputs of the discriminator $$D$$.

7. **Use better distribution similarity measure** [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric). See more in the next section. {{}}#wasserstein-gan-wgan

**Implementation**
[openai/improved-gan](https://github.com/openai/improved-gan)


## Wasserstein GAN (WGAN)

### What is Wasserstein Distance?

[Wasserstein Distance]() is a measure of the distance between two probability distributions.

It is also called **Earth Mover's distance** or EM distance, because informatlly it can be interpreted as moving piles of dirts that follows one probability distribution at a minimum cost to follow the other distribution. The cost is quantified by the amount of dirt moved times the moving distance.

https://en.wikipedia.org/wiki/Hungarian_algorithm
https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)

Let's first look at the case when the probability domain is discrete. For example, suppose we have two distributions $$P$$ and $$Q$$, each distribution has 4 piles of dirt and both have 10 shovelfuls of dirt in total. The numbers of shovelfuls in each dirt pile are:

$$
P_1 = 3, P_2 = 2, P_3 = 1, P_4 = 4\\
Q_1 = 1, Q_2 = 2, Q_3 = 4, Q_4 = 3
$$

In order to change $$P$$ to look like $$Q$$, as illustrated in Fig. x, we:
- First move 2 shovelfuls from $$P_1$$ to $$P_2$$ => $$(P_1, Q_1)$$ match up.
- Then move 2 shovelfuls from $$P_2$$ to $$P_3$$ => $$(P_2, Q_2)$$ match up.
- Finally move 1 shovelfuls from $$Q_3$$ to $$Q_4$$ => $$(P_3, Q_3)$$ and $$(P_4, Q_4)$$ match up. 

If we label the cost to pay to make $$P_i$$ and $$Q_i$$ match as $$\text{cost}_i$$, we would have $$\text{cost}_{i+1} = \text{cost}_i + P_i - Q_i$$ and in the example:

$$
\begin{aligned}
\text{cost}_0 &= 0\\
\text{cost}_1 &= 0 + 3 - 1 = 2\\
\text{cost}_2 &= 2 + 2 - 2 = 2\\
\text{cost}_3 &= 2 + 1 - 4 = -1\\
\text{cost}_4 &= -1 + 4 - 3 = 0
\end{aligned}
$$

Finally the Earth Mover's distance is $$W = \sum_{\vert \text{cost}_i \vert} = 5$$.


![EM distance for discrete case]({{ '/assets/images/EM_distance_discrete.png' | relative_url }})
{: class="center"}
*Fig. x. xxx.*


In continous cases, ...


$$
W(p_r, p_g) = \inf_{\gamma \sim \Pi(p_r, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\vert\vert x-y \vert\vert]
$$

In the formula above, $$\Pi(p_r, p_g)$$ is the set of all possible joint probability distributions. One joint distribution $$\gamma \in \Pi(p_r, p_g)$$ describes one dirt transport plan, same as the discrete example above, but in the continous probability space. Precisely $$\gamma(x, y)$$ states the percentage of dirt should be transported from point $$x$$ to $$y$$ so as to make $$x$$ follows the same probability distribution of $$y$$. That's why the marginal distribution over $$x$$ adds up to $$p_g$$, $$\sum_{x} \gamma(x, y) = p_g(y)$$ and vice versa $$\sum_{y} \gamma(x, y) = p_r(x)$$.

When treating $$x$$ as the starting point and $$y$$ as the destination, the total amount of dirt moved is $$\gamma(x, y)$$ and the travelling distance is $$\vert\vert x-y \vert\vert$$ and thus the cost is $$\gamma(x, y) \times \vert\vert x-y \vert\vert$$. The expected cost averaged across all the $$(x,y)$$ pairs can be easily computed, 

$$
\sum_{x, y} \gamma(x, y) \vert\vert x-y \vert\vert 
= \mathbb{E}_{x, y \sim \gamma} \vert\vert x-y \vert\vert
$$. 

Finally we take the minimum one among the costs of all dirt moving solutions. Here the $$\inf$$ ([infimum](https://en.wikipedia.org/wiki/Infimum_and_supremum), also known as *greatest lower bound*) indicates that we are only interested in the smallest cost. 


### Why Wasserstein is Better than JS or KL Divergence?

Even when two distributions are located in lower dimensionals manifolds without overlaps, Wasserstein distance can still provide a meaningful and smooth representation of the distance in-between.

The WGAN paper exampified the idea with a simple example.
Suppose we have two probability distributions, $$P$$ and $$Q$$:

$$
\forall (x, y) \in P, x = 0 \text{ and } y \sim U(0, 1)\\
\forall (x, y) \in Q, x = \theta, 0 \leq \theta \leq 1 \text{ and } y \sim U(0, 1)\\
$$

![Simple example]({{ '/assets/images/wasserstein_simple_example.png' | relative_url }})
{: style="width: 480px;" class="center"}
*Fig. x. There is no overlap between $$P$$ and $$Q$$.*

In this case, when $$\theta \neq 0$$:

$$
\begin{aligned}
D_{KL}(P \vert\vert Q) &= \sum_{x=0, y \sim U(0, 1)} 1 \cdot \log\frac{1}{0} = +\infty \\
D_{KL}(Q \vert\vert P) &= \sum_{x=\theta, y \sim U(0, 1)} 1 \cdot \log\frac{1}{0} = +\infty \\
D_{JS}(P, Q) &= \frac{1}{2}(\sum_{x=0, y \sim U(0, 1)} 1 \cdot \log\frac{1}{1/2} + \sum_{x=0, y \sim U(0, 1)} 1 \cdot \log\frac{1}{1/2}) = \log 2\\
W(P, Q) &= |\theta|
\end{aligned}
$$

But when $$\theta = 0$$, two distributions are fully overlapped:

$$
\begin{aligned}
D_{KL}(P \| Q) &= D_{KL}(Q \| P) = D_{JS}(P, Q) = 0\\
W(P, Q) &= 0 = \lvert \theta \rvert
\end{aligned}
$$

Only Wasserstein metric provides a smooth measure, which is super helpful for a smooth and stable learning process with gradient descents.

### Use Wasserstein Distance as GAN Loss Function

It is intractable to exhaust all the possible joint distributions in $$\Pi(p_r, p_g)$$ to compute $$\inf_{\gamma \sim \Pi(p_r, p_g)}$$. Thus the authors proposed a smart transformation of the formula based on the Kantorovich-Rubinstein duality to:

$$
W(p_r, p_g) = \frac{1}{K} \sup_{\| f \|_L \leq K} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]
$$

where $$\sup$$ ([supremum](https://en.wikipedia.org/wiki/Infimum_and_supremum)) is the opposite of $$inf$$ (infimum); we want to measure the least upper bound or, in even simpler words, the maximum value.


**Lipschitz Continuity**

The function $$f$$ is demanded to satisfy $$\| f \|_L \leq K$$, meaning it should be [K-Lipschitz continuous](https://en.wikipedia.org/wiki/Lipschitz_continuity).

A real-valued function $$f: \mathbb{R} \rightarrow \mathbb{R}$$ is called $$K$$-Lipschitz continuous if there exists a real constant $$K \geq 0$$ such that, for all $$x_1, x_2 \in \mathbb{R}$$,

$$\lvert f(x_1) - f(x_2) \rvert \leq K \lvert x_1 - x_2 \rvert$$

Functions that are everywhere continuously differentiable is Lipschitz continuous, because the derivative, estimated as $$\frac{\lvert f(x_1) - f(x_2) \rvert}{\lvert x_1 - x_2 \rvert}$$, has bounds. However, a Lipschitz continuous function may not be everywhere differentiable, such as $$f(x) = \lvert x \rvert$$.

Describing how the transformation happens on the Wasserstein distance formula is worthy of a post by itself, so I skip the details here. If you are interested in how to compute Wasserstein metric using linear programming, what is the Kantorovich-Rubinstein Duality and how to transfer Wasserstein metric into its dual form according to the Kantorovich-Rubinstein Duality, read this [awesome post](https://vincentherrmann.github.io/blog/wasserstein/) by Vincent Herrmann.

Suppose this function $$f$$ comes from a family of K-Lipschitz continuous functions, $$\{ f_w \}_{w \in W}$$, parameterized by $$w$$. In this modified Wasserstein-GAN, let us use the "discriminator" model to learn $$w$$ and set the loss function as the Wasserstein distance:

$$
L(p_r, p_g) = W(p_r, p_g) = \max_{w \in W} \mathbb{E}_{x \sim p_r}[f_w(x)] - \mathbb{E}_{z \sim p_r(z)}[f_w(g_\theta(z))]
$$

So far the "discriminator" is not a real critic of tellling the fake samples from the real ones any more. Instead it is trained to learn a $$K$$-Lipschitz continuous function to help compute Wasserstein distance. As the loss function decreases in the training, the Wasserstein distance gets smaller and the generator model $$g_\theta$$'s output turns closer to the real data distribution.

One big problem is to maintain the state of the $$K$$-Lipschitz continuity of $$f_w$$ during training. The paper presents a simple but very practical trick: After every gradient update, clamp the weights $$w$$ to a small window, such as $$[-0.01, 0.01]$$, resulting in a compact parameter space $$W$$ and a "critic" function $$f_w$$ with upper bounds to preserve the Lipschitz continuity.



![Simple example]({{ '/assets/images/WGAN_algorithm.png' | relative_url }})
{: style="width: 640px;" class="center"}
*Fig. x. Algorithm.*


Compared to the original GAN algorithm, the WGAN undertakes the following changes:
1. After every gradient update on the critic function, clamp the weights to a small fixed range, $$[-c, c]$$.
2. Use a new loss function derived from the Wasserstein distance, no logarithm any more. The "discriminator" model does not play as a real critic but a helper for estimating the Wasserstein metric between real and generated data distribution.
3. Emprically the authors recommended [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) optimizer on the critic, rather than a momentum based optimizer such as [Adam](https://arxiv.org/abs/1412.6980v8) which could cause instability in the model training.


---

Sadly, Wasserstein GAN is not perfect. Even the authors in the original WGAN mentioned that *"Weight clipping is a clearly terrible way to enforce a Lipschitz constraint"*. WGAN still suffers from unstable training, slow convergence after weight clipping (when clipping window is too large), and vanishing gradients (when clipping window is too small).

Some improvement, precisely replacing weight clipping with **gradient penalty**, has been discussed in [Gulrajani et al. 2017](https://arxiv.org/pdf/1704.00028.pdf). I will leave this to future posts.


## Example: Creat New Pokemons!

GAN vs WGAN
![Generative adversarial network]({{ '/assets/images/pokemon-GAN.png' | relative_url }})
{: class="center"}
*Fig. x. Pokemon!*


## References

[1] Goodfellow, Ian, et al. "Generative adversarial nets." NIPS, 2014. https://arxiv.org/pdf/1406.2661.pdf

[2] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. "Improved techniques for training gans." In Advances in Neural Information Processing Systems, pp. 2234-2242. 2016. http://papers.nips.cc/paper/6125-improved-techniques-for-training-gans.pdf

[3] Martin Arjovsky and Léon Bottou. "Towards principled methods for training generative adversarial networks." arXiv preprint arXiv:1701.04862 (2017). https://arxiv.org/pdf/1701.04862.pdf

[4] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville. Improved training of wasserstein gans. arXiv preprint arXiv:1704.00028. 2017 Mar 31. https://arxiv.org/pdf/1704.00028.pdf

[4] https://zhuanlan.zhihu.com/p/25071913?from=timeline&isappinstalled=0

http://robotics.stanford.edu/~scohen/research/emdg/emdg.html

https://vincentherrmann.github.io/blog/wasserstein/

https://github.com/jiamings/wgan
