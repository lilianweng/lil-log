---
layout: post
comments: false
title: "Anatomize Deep Learning with Information Theory"
date: 2017-09-28 23:42:00
---

> This post is a summary of Prof Naftali Tishby's recent talk on ["Information Theory in Deep Learning"](https://youtu.be/bLqJHjXihK8). It presented how to apply the information theory to study the growth and transformation of deep neural networks during training.

<!--more-->

Recently I watched the talk ["Information Theory in Deep Learning"](https://youtu.be/bLqJHjXihK8) by Prof Naftali Tishby and found it very interesting. He presented how to apply the information theory to study the growth and transformation of deep neural networks during training. Using the Information Bottleneck (IB) theory, he proposed a new learning bound for deep neural networks, for which the traditional learning theory fails due to the exponentially large amount of parameters. Another keen observation is that the training of a deep neural network has two phases: First, the network is trained to fully represent the input data and minimize the generalization error; then, it learns to forget the irrelevant details by compressing the representation of the input data.

Most of the materials in this post are from Prof Tishby’s talk and [related papers].



## Basic Concepts

[Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)

A Markov process is a stochastic process is ["memoryless"](http://mathworld.wolfram.com/Memoryless.html) (also called "Markov Property"). A Markov chain is a type of Markov process containing multiple discrete states. That is being said, the conditional probability of future states of the process is only determined by the current state and does not depend on the past states.


[Kullback–Leibler (KL) Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

KL divergence measures how one probability distribution $$p$$ diverges from a second expected probability distribution $$q$$. It is asymmetric.

$$
\begin{aligned}
D_{KL}(p \| q) &= \sum_x p(x) \log \frac{p(x)}{q(x)} dx \\
 &= - \sum_x p(x)\log q(x) + \sum_x p(x)\log p(x) \\
 &= H(P, Q) - H(P)
\end{aligned}
$$

$$D_{KL}$$ achieves the minimum zero when $$p(x)$$ == $$q(x)$$ everywhere.

[Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)

Mutual information measures the mutual dependence between two variables. It quantifies the "amount of information" obtained about one random variable through the other random variable. Mutual information is symmetric.

$$
\begin{aligned}
I(X;Y) &= D_{KL}[p(x,y) \| p(x)p(y)] \\
 &= \sum_{x \in X, y \in Y} p(x, y) \log(\frac{p(x, y)}{p(x)p(y)}) \\
 &= \sum_{x \in X, y \in Y} p(x, y) \log(\frac{p(x|y)}{p(x)}) \\ 
 &= H(X) - H(X|Y) \\
\end{aligned}
$$


[Data Processing Inequality (DPI)](https://en.wikipedia.org/wiki/Data_processing_inequality)

For any markov chain: $$X \to Y \to Z$$, we would have $$I(X; Y) > I(X; Z)$$.

A deep neural network (DNN) can be viewed as a Markov chain, and thus when we are moving down the layers of a DNN, from the input layer to hidden layers to the output one, the mutual information between the layer and the input can only decrease.


[Reparametrization invariance](https://en.wikipedia.org/wiki/Parametrization#Parametrization_invariance)

For two invertible functions $$\phi$$, $$\psi$$, the mutual information still holds: $$I(X; Y) = I(\phi(X); \psi(Y))$$.

For example, if we shuffle the weights in one layer of DNN, it would not affect the mutual information.



## Deep Neural Networks as Markov Chains

The training data for a deep neural network is a set of sampled observations from the joint distribution of $$X$$ and $$Y$$. The input variable $$X$$ and weights of hidden layers are all high-dimensional random variable. The ground truth target $$Y$$ and the predicted value $$\hat{Y}$$ are random variables of smaller dimensions (in classification setting).



![DNN structure]({{ '/assets/images/ib-dnn-structure.png' | relative_url }})
{: style="width: 460px;" class="center"}
*Fig. 1. The structure of a deep neural network, which consists of the target label $$Y$$, input layer $$X$$, hidden layers $$h_1, \dots, h_m$$ and the final prediction $$\hat{Y}$$. (Image source: 
[Tishby and Zaslavsky, 2015](https://arxiv.org/pdf/1503.02406.pdf))*


If we label the hidden layers of a DNN as $$h_1, h_2, \dots, h_m$$ as in Fig. 1, we can view each layer as one state of a Markov Chain: $$ h_i \to h_{i+1}$$. According to DPI, we would have:

$$
H(X) \geq I(X; h_1) \geq I(X; h_2) \geq \dots \geq I(X; h_m) \geq I(X; \hat{Y}) \\
I(X; Y) \geq I(h_1; Y) \geq I(h_2; Y) \geq \dots \geq I(h_m; Y) \geq I(\hat{Y}; Y)
$$

A DNN is designed to learn how to describe $$X$$ to predict $$Y$$ and eventually, to compress $$X$$ to only hold the information related to $$Y$$. Tishby describes this processing as *"successive refinement of relevant information"*.


### Information Plane Theorem

A DNN has successive internal representations of $$X$$, a set of hidden layers $$\{T_i\}$$. The *information plane* theorem characterizes each layer by its encoder and decoder information.

Precisely, in an information plane plot:
- **X-axis**: The sample complexity of $$T_i$$ is determined by the encoder mutual information $$I(X; T_i)$$. Sample complexity refers to how many samples you need to achieve certain accuracy and generalization.
- **Y-axis**: The accuracy (generalization error) is determined by the decoder mutual information $$I(T_i; Y)$$.


![Information plane]({{ '/assets/images/ib-information-plane.png' | relative_url }})
{: style="width: 640px;" class="center"}
*Fig. 2. Plots of $$I(X; T)$$ vs $$I(T; Y)$$ of hidden layers of 50 experiments. Layers are color-coders, there are three snapshots at the initial epoch, 400 epochs and 9000 epochs. (Image source: [Shwartz-Ziv and Tishby, 2017](https://arxiv.org/pdf/1703.00810.pdf))*


Each dot in Fig. 2. identifies encoder/ decoder measures of one hidden layers of one network simulation (no regularization is applied; no weights decay, no dropout, etc.). They move up as expected, because the knowledge about the true labels is increasing (accuracy increases). At the early stage, the hidden layers learns a lot about the input $$X$$, but later they start to compress and to forget something about the input. Tishby believes that "the most important part of learning is actually forgetting". Check out this [nice video](https://youtu.be/P1A1yNsxMjc) that demonstrates how the mutual information measures of layers are changing in time.



![Information plane merged]({{ '/assets/images/ib-information-plane-merged.png' | relative_url }})
{: style="width: 400px;" class="center"}
*Fig. 3. Here is an aggregated view of Fig 2. (Image source: [Tishby’ talk 15:15](https://youtu.be/bLqJHjXihK8?t=15m15s))*


The compression happens after the generalization error becomes very small.


### Two Optimization Phases
                    
Tracking the normalized mean and standard deviation of the weights of each layer in the epoch time can also reveal two phases of the training process.


![mean and std]({{ '/assets/images/ib-mean-variation.png' | relative_url }})
{: style="width: 480px;" class="center"}
*Fig. 4. The normalized mean and standard deviation of each layer's weights as a function of training epochs. Different layers are color-coded. (Image source: [Shwartz-Ziv and Tishby, 2017](https://arxiv.org/pdf/1703.00810.pdf))*

Among early epochs, the mean values are three magnitude larger than the standard deviations. After a sufficient number of epochs, the error saturates and the standard deviation becomes much more noisy afterwards. The further a layer is away from the output, the noisier it gets, because the noises can get amplified and accumulated through the back-prop process (not the width of the layer).




## Learning Theory

### "Old" Generalization Bounds

The generalization bounds defined by the classic learning theory is:

$$
\epsilon^2 < \frac{\log|H_\epsilon| + \log{1/\delta}}{2m}
$$

- $$\epsilon$$: generalization error; a measure of how accurately an algorithm is able to predict outcome values for previously unseen data.
- $$H_\epsilon$$: $$\epsilon$$-cover of the hypothesis class. Typically we assume $$\vert H_\epsilon \vert \sim (1/\epsilon)^d$$.
- $$\delta$$: confidence.
- $$m$$: the number of training samples.
- $$d$$: the VC dimension of the hypothesis.

This bound states that the generalization error is bounded by the training error plus a function of the hypothesis space size and the dataset size. The bigger the hypothesis space gets, the bigger the generalization error becomes. I recommend these two tutorials on ML theory, [part1](https://mostafa-samir.github.io/ml-theory-pt1/) and [part2](https://mostafa-samir.github.io/ml-theory-pt2/), if you are interested in reading more on generalization bounds.

However, it does not work for deep learning. The larger a network is, the more parameters it needs to learn. With this generalization bounds, larger networks (larger $$d$$) would have worse bounds. This is contrary to the intuition that larger networks are able to achieve better performance with higher expressivity. 


### "New" Input compression bound

To solve this counterintuitive observation, Tishby et al. proposed a new input compression bound for DNN.

First let's have $$T_\epsilon$$ as an $$\epsilon$$-partition of the input variable $$X$$. This partition compresses the input with homogeneity to the labels with respect to the labels into small cells that can together cover the whole input space. If the prediction outputs binary values, we can replace the cardinality of the hypothesis, $$\vert H_\epsilon \vert$$, with $$2^{\vert T_\epsilon \vert}$$.

$$
|H_\epsilon| \sim 2^{|X|} \to 2^{|T_\epsilon|}
$$

When $$X$$ is large, the size of $$X$$ is approximately $$2^{H(X)}$$. Each cell in the $$\epsilon$$-partition is of size $$2^{H(X \vert T_\epsilon)}$$. Therefore we have $$\vert T_\epsilon \vert \sim \frac{2^{H(X)}}{2^{H(X \vert T_\epsilon)}} = 2^{I(T_\epsilon; X)}$$

then we have the input compression bound as:

$$
\epsilon^2 < \frac{2^{I(T_\epsilon; X)} + \log{1/\delta}}{2m}
$$



![IB bounds]({{ '/assets/images/ib-bound.png' | relative_url }})
{: style="width: 480px;" class="center"}
*Fig. 5. The black line is the optimal achievable information bottleneck (IB) limit. The red line corresponds to the upper bound on the out-of-sample IB distortion (mutual information with $$Y$$), when training from a finite sample. (Recreated based on [Tishby’ talk 24:50](https://youtu.be/bLqJHjXihK8?t=24m56s))*


## Network Size and Training Data Size


### The Benefit of More Hidden Layers

More layers take much fewer training epochs for good generalization. Since the network has to spend a lot of time on compressing the representation, having more hidden layers can give us computational benefits and speed up the compression relaxation time.


![Layers]({{ '/assets/images/ib-layers.png' | relative_url }})
{: style="width: 640px;" class="center"}
*Fig. 6. The optimization time is much shorter with more hidden layers. Converging faster. (Image source: [Shwartz-Ziv and Tishby, 2017](https://arxiv.org/pdf/1703.00810.pdf))*


Compression through stochastic relaxation: According to the [diffusion equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation), the relaxation time of layer $$k$$ is proportional to the exponential of this layer's compression amount $$\Delta S_k$$: $$\Delta t_k \sim \exp(\Delta S_k)$$. We can compute the layer compression as $$\Delta S_k = I(X; T_k) - I(X; T_{k-1})$$.  Because $$\exp(\sum_k \Delta S_k) \geq \sum_k \exp(\Delta S_k)$$, we would expect an exponential decrease in training epoch times with more hidden layers (larger $$k$$).



### The Benefit of More Training Samples

Fitting larger training data requires more information captured by the hidden layers. With increased training data size, the decoder mutual information (recall that this is directly related to the generalization error), $$I(T; Y)$$, is pushed up and gets closer to the theoretical information bottleneck bound. Tishby emphasized that It is the mutual information, not layer size or VC-dim that determines generalization, different from standard theories.



![Training size]({{ '/assets/images/ib-training-size.png' | relative_url }})
{: style="width: 440px;" class="center"}
*Fig. 7. The training data of different sizes is color-coded. The information plane of multiple converged networks are plotted. (Image source: [Shwartz-Ziv and Tishby, 2017](https://arxiv.org/pdf/1703.00810.pdf))*


## References

[1] Naftali Tishby. [Information Theory of Deep Learning](https://youtu.be/bLqJHjXihK8)

[2] [Machine Learning Theory - Part 1: Introduction](https://mostafa-samir.github.io/ml-theory-pt1/)

[3] [Machine Learning Theory - Part 2: Generalization Bounds](https://mostafa-samir.github.io/ml-theory-pt2/)

[4] [New Theory Cracks Open the Black Box of Deep Learning](https://www.quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921/) by Quanta Magazine.

[5] Naftali Tishby and Noga Zaslavsky. ["Deep learning and the information bottleneck principle."](https://arxiv.org/pdf/1503.02406.pdf) IEEE Information Theory Workshop (ITW), 2015.

[6] Ravid Shwartz-Ziv and Naftali Tishby. ["Opening the Black Box of Deep Neural Networks via Information."](https://arxiv.org/pdf/1703.00810.pdf) arXiv preprint arXiv:1703.00810, 2017.




