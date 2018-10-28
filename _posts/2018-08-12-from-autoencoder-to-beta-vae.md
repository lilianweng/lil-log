---
layout: post
comments: true
title: "From Autoencoder to Beta-VAE"
date: 2018-08-12 10:18:00
tags: review autoencoder generative
image: "vae-gaussian.png"
---

> Autocoders are a family of neural network models aiming to learn compressed latent variables of high-dimensional data. Starting from the basic autocoder model, this post reviews several variations, including denoising, sparse, and contractive autoencoders, and then Variational Autoencoder (VAE) and its modification beta-VAE.


<!--more-->


Autocoder is invented to reconstruct high-dimensional data using a neural network model with a narrow bottleneck layer in the middle (oops, this is probably not true for [Variational Autoencoder](#vae-variational-autoencoder), and we will investigate it in details in later sections). A nice byproduct is dimension reduction: the bottleneck layer captures a compressed latent encoding. Such a low-dimensional representation can be used as en embedding vector in various applications (i.e. search), help data compression, or reveal the underlying data generative factors. 


{: class="table-of-content"}
* TOC
{:toc}


## Notation

{: class="info"}
| Symbol | Mean |
| ---------- | ---------- |
| $$\mathcal{D}$$ | The dataset, $$\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(n)} \}$$, contains $$n$$ data samples; $$\vert\mathcal{D}\vert =n $$. |
| $$\mathbf{x}^{(i)}$$ | Each data point is a vector of $$d$$ dimensions, $$\mathbf{x}^{(i)} = [x^{(i)}_1, x^{(i)}_2, \dots, x^{(i)}_d]$$. |
| $$\mathbf{x}$$ | One data sample from the dataset, $$\mathbf{x} \in \mathcal{D}$$. |
| $$\mathbf{x}’$$| The reconstructed version of $$\mathbf{x}$$. |
| $$\tilde{\mathbf{x}}$$ | The corrupted version of $$\mathbf{x}$$. |
| $$\mathbf{z}$$ | The compressed code learned in the bottleneck layer. |
| $$a_j^{(l)}$$ | The activation function for the $$j$$-th neuron in the $$l$$-th hidden layer. |
| $$g_{\phi}(.)$$ | The **encoding** function parameterized by $$\phi$$. |
| $$f_{\theta}(.)$$ | The **decoding** function parameterized by $$\theta$$. |
| $$q_{\phi}(\mathbf{z}\vert\mathbf{x})$$ |Estimated posterior probability function, also known as **probabilistic encoder**.  |
| $$p_{\theta}(\mathbf{x}\vert\mathbf{z})$$ | Likelihood of generating true data sample given the latent code, also known as **probabilistic decoder**. |
| ---------- | ---------- |



## Autoencoder

**Autoencoder** is a neural network designed to learn an identity function in an unsupervised way  to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation. The idea was originated in [the 1980s](https://en.wikipedia.org/wiki/Autoencoder), and later promoted by the seminal paper by [Hinton & Salakhutdinov, 2016](https://pdfs.semanticscholar.org/c50d/ca78e97e335d362d6b991ae0e1448914e9a3.pdf).

It consists of two networks:
- *Encoder* network: It translates the original high-dimension input into the latent low-dimensional code. The input size is larger than the output size.
- *Decoder* network: The decoder network recovers the data from the code, likely with larger and larger output layers.


![Autoencoder architecture]({{ '/assets/images/autoencoder-architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. Illustration of autoencoder model architecture.*


The encoder network essentially accomplishes the [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), just like how we would use Principal Component Analysis (PCA) or Matrix Factorization (MF) for. In addition, the autoencoder is explicitly optimized for the data reconstruction from the code. A good intermediate representation not only can capture latent variables, but also benefits a full [decompression](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html) process.

The model contains an encoder function $$g(.)$$ parameterized by $$\phi$$ and a decoder function $$f(.)$$ parameterized by $$\theta$$. The low-dimensional code learned for input $$\mathbf{x}$$ in the bottleneck layer is $$\mathbf{z} = $$ and the reconstructed input is $$\mathbf{x}' = f_\theta(g_\phi(\mathbf{x}))$$.

The parameters $$(\theta, \phi)$$ are learned together to output a reconstructed data sample same as the original input, $$\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$$, or in other words, to learn an identity function. There are various metrics to quantify the difference between two vectors, such as cross entropy when the activation function is sigmoid, or as simple as MSE loss:

$$
L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2
$$


## Denoising Autoencoder

Since the autoencoder learns the identity function, we are facing the risk of “overfitting” when there are more network parameters than the number of data points. 

To avoid overfitting and improve the robustness, **Denoising Autoencoder** (Vincent et al. 2008) proposed a modification to the basic autoencoder. The input is partially corrupted by adding noises to or masking some values of the input vector in a stochastic manner, $$\tilde{\mathbf{x}} \sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$. Then the model is trained to recover the original input (**Note: Not the corrupt one!**).


$$
\begin{aligned}
\tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\
L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2
\end{aligned}
$$


where $$\mathcal{M}_\mathcal{D}$$ defines the mapping from the true data samples to the noisy or corrupted ones.


![Denoising autoencoder architecture]({{ '/assets/images/denoising-autoencoder-architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. Illustration of denoising autoencoder model architecture.*


This design is motivated by the fact that humans can easily recognize an object or a scene even the view is partially occluded or corrupted. To “repair” the partially destroyed input, the denoising autoencoder has to discover and capture relationship between dimensions of input in order to infer missing pieces. 

For high dimensional input with high redundancy, like images, the model is likely to depend on evidence gathered from a combination of many input dimensions to recover the denoised version (sounds like the [attention]({{ site.baseurl }}{% post_url 2018-06-24-attention-attention %}) mechanism, right?) rather than to overfit one dimension. This builds up a good foundation for learning *robust* latent representation.

The noise is controlled by a stochastic mapping $$\mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$$, and it is not specific to a particular type of corruption process (i.e. masking noise, Gaussian noise, salt-and-pepper noise, etc.). Naturally the corruption process can be equipped with prior knowledge

In the experiment of the original DAE paper, the noise is applied in this way: a fixed proportion of input dimensions are selected at random and their values are forced to 0. Sounds a lot like dropout, right? Well, the denoising autoencoder was proposed in 2008, 4 years before the dropout paper ([Hinton, et al. 2012](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)) ;)

<!-- 
**Stacked Denoising Autoencoder**: In the old days when it was still hard to train deep neural networks, stacking denoising autoencoders was a way to build deep models ([Vincent et al., 2010](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)). The denoising autoencoders are trained layer by layer. Once one layer has been trained, it is fed with clean, uncorrupted inputs to learn the encoding in the next layer.


![Stacking denoising autoencoder]({{ '/assets/images/stacking-dae.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 3. Stacking denoising autoencoders. (Image source: [Vincent et al., 2010](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf))*
-->


## Sparse Autoencoder

**Sparse Autoencoder** applies a "sparse" constraint on the hidden unit activation to avoid overfitting and improve robustness. It forces the model to only have a small number of hidden units being activated at the same time, or in other words, one hidden neuron should be inactivate most of time.

Recall that common [activation functions](http://cs231n.github.io/neural-networks-1/#actfun) include sigmoid, tanh, relu, leaky relu, etc. A neuron is activated when the value is close to 1 and inactivate with a value close to 0.

Let’s say there are $$s_l$$ neurons in the $$l$$-th hidden layer and the activation function for the $$j$$-th neuron in this layer is labelled as $$a^{(l)}_j(.)$$, $$j=1, \dots, s_l$$. The fraction of activation of this neuron $$\hat{\rho}_j$$ is expected to be a small number $$\rho$$, known as *sparsity parameter*; a common config is $$\rho = 0.05$$.


$$
\hat{\rho}_j^{(l)} = \frac{1}{n} \sum_{i=1}^n [a_j^{(l)}(\mathbf{x}^{(i)})] \approx \rho
$$

This constraint is achieved by adding a penalty term into the loss function. The KL-divergence $$D_\text{KL}$$ measures the difference between two Bernoulli distributions, one with mean $$\rho$$ and the other with mean $$\hat{\rho}_j^{(l)}$$. The hyperparameter $$\beta$$ controls how strong the penalty we want to apply on the sparsity loss.


$$
\begin{aligned}
L_\text{SAE}(\theta) 
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} D_\text{KL}(\rho \| \hat{\rho}_j^{(l)}) \\
&= L(\theta) + \beta \sum_{l=1}^L \sum_{j=1}^{s_l} \rho\log\frac{\rho}{\hat{\rho}_j^{(l)}} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j^{(l)}}
\end{aligned}
$$


![KL divergence]({{ '/assets/images/kl-metric-sparse-autoencoder.png' | relative_url }})
{: style="width: 80%;" class="center"}
*Fig. 4. The KL divergence between a Bernoulli distribution with mean $$\rho=0.25$$ and a Bernoulli distribution with mean $$0 \leq \hat{\rho} \leq 1$$.*


**$$k$$-Sparse Autoencoder**

In $$k$$-Sparse Autoencoder ([Makhzani and Frey, 2013](https://arxiv.org/abs/1312.5663)), the sparsity is enforced by only keeping the top k highest activations in the bottleneck layer with linear activation function. 
First we run feedforward through the encoder network to get the compressed code: $$\mathbf{z} = g(\mathbf{x})$$.
Sort the values  in the code vector $$\mathbf{z}$$. Only the k largest values are kept while other neurons are set to 0. This can be done in a ReLU layer with an adjustable threshold too. Now we have a sparsified code: $$\mathbf{z}’ = \text{Sparsify}(\mathbf{z})$$.
Compute the output and the loss from the sparsified code, $$L = \|\mathbf{x} - f(\mathbf{z}') \|_2^2$$.
And, the back-propagation only goes through the top k activated hidden units!


![k-sparse autoencoder]({{ '/assets/images/k-sparse-autoencoder.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 5. Filters of the k-sparse autoencoder for different sparsity levels k, learnt from MNIST with 1000 hidden units.. (Image source: [Makhzani and Frey, 2013](https://arxiv.org/abs/1312.5663))*



## Contractive Autoencoder

Similar to sparse autoencoder, **Contractive Autoencoder** ([Rifai, et al, 2011](http://www.icml-2011.org/papers/455_icmlpaper.pdf)) encourages the learned representation to stay in a contractive space for better robustness. 

It adds a term in the loss function to penalize the representation being too sensitive to the input,  and thus improve the robustness to small perturbations around the training data points. The sensitivity is measured by the Frobenius norm of the Jacobian matrix of the encoder activations with respect to the input:


$$
\|J_f(\mathbf{x})\|_F^2 = \sum_{ij} \Big( \frac{\partial h_j(\mathbf{x})}{\partial x_i} \Big)^2
$$

where $$h_j$$ is one unit output in the compressed code $$\mathbf{z} = f(x)$$. 

This penalty term is the sum of squares of all partial derivatives of the learned encoding with respect to input dimensions. The authors claimed that empirically this penalty was found to  carve a representation that corresponds to a lower-dimensional non-linear manifold, while staying more invariant to majority directions orthogonal to the manifold.



## VAE: Variational Autoencoder

The idea of **Variational Autoencoder** ([Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)), short for **VAE**, is actually less similar to all the autoencoder models above, but deeply rooted in the methods of variational bayesian and graphical model.

Instead of mapping the input into a *fixed* vector, we want to map it into a distribution. Let’s label this distribution as $$p_\theta$$, parameterized by $$\theta$$.  The relationship between the data input $$\mathbf{x}$$ and the latent encoding vector $$\mathbf{z}$$ can be fully defined by:
- Prior $$p_\theta(\mathbf{z})$$
- Likelihood $$p_\theta(\mathbf{x}\vert\mathbf{z})$$
- Posterior $$p_\theta(\mathbf{z}\vert\mathbf{x})$$


Assuming that we know the real parameter $$\theta^{*}$$ for this distribution. In order to generate a sample that looks like a real data point $$\mathbf{x}^{(i)}$$, we follow these steps:
1. First, sample a $$\mathbf{z}^{(i)}$$ from a prior distribution $$p_{\theta^*}(\mathbf{z})$$. 
2. Then a value $$\mathbf{x}^{(i)}$$ is generated from a conditional distribution $$p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$$.


The optimal parameter $$\theta^{*}$$ is the one that maximizes the probability of generating real data samples:


$$
\theta^{*} = \arg\max_\theta \prod_{i=1}^n p_\theta(\mathbf{x}^{(i)})
$$

Commonly we use the log probabilities to convert the product on RHS to a sum:


$$
\theta^{*} = \arg\max_\theta \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)})
$$

Now let’s update the equation to better demonstrate the data generation process so as to involve the encoding vector:


$$
p_\theta(\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}\vert\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z} 
$$

Unfortunately it is not easy to compute $$p_\theta(\mathbf{x}^{(i)})$$ in this way, as it is very expensive to check all the possible values of $$\mathbf{z}$$ and sum them up. To narrow down the value space to facilitate faster search, we would like to introduce a new approximation function to output what is a likely code given an input $$\mathbf{x}$$, $$q_\phi(\mathbf{z}\vert\mathbf{x})$$, parameterized by $$\phi$$.


![Distributions in VAE]({{ '/assets/images/VAE-graphical-model.png' | relative_url }})
{: style="width: 80%;" class="center"}
*Fig. 6. The graphical model involved in Variational Autoencoder.  Solid lines denote the generative distribution $$p_\theta(.)$$ and dashed lines denote the distribution $$q_\phi (\mathbf{z}\vert\mathbf{x})$$ to approximate the intractable posterior $$p_\theta (\mathbf{z}\vert\mathbf{x})$$.*

Now the structure looks a lot like an autoencoder:
- The conditional probability $$p_\theta(\mathbf{x} \vert \mathbf{z})$$ defines a generative model, similar to the decoder $$f_\theta(\mathbf{x} \vert \mathbf{z})$$ introduced above. $$p_\theta(\mathbf{x} \vert \mathbf{z})$$ is also known as *probabilistic decoder*. 
- The approximation function $$q_\phi(\mathbf{z} \vert \mathbf{x})$$ is the *probabilistic encoder*, playing a similar role as $$g_\phi(\mathbf{z} \vert \mathbf{x})$$ above.



### Loss Function

The estimated posterior $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ should be very close to the real one $$p_\theta(\mathbf{z}\vert\mathbf{x})$$. We can use [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) to quantify the distance between these two distributions. KL divergence $$D_\text{KL}(X\|Y)$$ measures how much information is lost if the distribution Y is used to represent X.

In our case we want to minimize $$D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )$$ with respect to $$\phi$$.

But why use $$D_\text{KL}(q_\phi \| p_\theta)$$ (reversed KL) instead of $$D_\text{KL}(p_\theta \| q_\phi)$$ (forward KL)? Eric Jang has a great explanation in his [post](https://blog.evjang.com/2016/08/variational-bayes.html) on Bayesian Variational methods. As a quick recap:


![Forward vs reversed KL]({{ '/assets/images/forward_vs_reversed_KL.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Forward and reversed KL divergence have different demands on how to match two distributions. (Image source: [blog.evjang.com/2016/08/variational-bayes.html](https://blog.evjang.com/2016/08/variational-bayes.html))*

- Forward KL divergence: $$D_\text{KL}(P\|Q) = \mathbb{E}_{z\sim P(z)} \log\frac{P(z)}{Q(z)}$$; we have to ensure that Q(z)>0 wherever P(z)>0. The optimized variational distribution $$q(z)$$ has to cover over the entire $$p(z)$$.
- Reversed KL divergence: $$D_\text{KL}(Q\|P) = \mathbb{E}_{z\sim Q(z)} \log\frac{Q(z)}{P(z)}$$; minimizing the reversed KL divergence squeezes the $$Q(z)$$ under $$P(z)$$.


Let's now expand the equation:

$$
\begin{aligned}
& D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z \vert x) = p(z, x) / p(x)} \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} & \\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }\int q(z \vert x) dz = 1}\\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{x}\vert\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z, x) = p(x \vert z) p(z)} \\
&=\log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z} \vert \mathbf{x})}[\log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z})} - \log p_\theta(\mathbf{x} \vert \mathbf{z})] &\\
&=\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) &
\end{aligned}
$$


So we have:

$$
D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) =\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z})
$$


Once rearrange the left and right hand side of the equation,

$$
\log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}))
$$

The LHS of the equation is exactly what we want to maximize when learning the true distributions: we want to maximize the (log-)likelihood of generating real data (that is $$\log p_\theta(\mathbf{x})$$) and also minimize the difference between the real and estimated posterior distributions (the term $$D_\text{KL}$$ works like a regularizer).  Note that $$p_\theta(\mathbf{x})$$ is fixed with respect to $$q_\phi$$.

The negation of the above defines our loss function:

$$
\begin{aligned}
L_\text{VAE}(\theta, \phi) 
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )\\
&= - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} p_\theta(\mathbf{x}\vert\mathbf{z}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}) ) \\
\theta^{*}, \phi^{*} &= \arg\min_{\theta, \phi} L_\text{VAE}
\end{aligned}
$$

In Variational Bayesian methods, this loss function is known as the *variational lower bound*, or *evidence lower bound*. The “lower bound” part in the name comes from the fact that KL divergence is always non-negative and thus $$L_\text{VAE}$$ is the lower bound of $$\log p_\theta (\mathbf{x})$$. 

$$
-L_\text{VAE} = \log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) \leq \log p_\theta(\mathbf{x})
$$

Therefore by minimizing the loss, we are maximizing the lower bound of the probability of generating real data samples.


### Reparameterization Trick

The expectation term in the loss function invokes generating samples from $$\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$$. Sampling is a stochastic process and therefore we cannot backpropagate the gradient. To make it trainable, the reparameterization trick is introduced: It is often possible to express the random variable $$\mathbf{z}$$ as a deterministic variable $$\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$$, where $$\boldsymbol{\epsilon}$$ is an auxiliary independent random variable, and the transformation function $$\mathcal{T}_\phi$$ parameterized by $$\phi$$ converts $$\boldsymbol{\epsilon}$$ to $$\mathbf{z}$$.

For example, a common choice of the form of $$q_\phi(\mathbf{z}\vert\mathbf{x})$$ is a multivariate Gaussian with a diagonal covariance structure:


$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; Reparameterization trick.}}
\end{aligned}
$$

where $$\odot$$ refers to element-wise product.


![Reparameterization trick]({{ '/assets/images/reparameterization-trick.png' | relative_url }})
{: style="width: 80%;" class="center"}
*Fig. 8. Illustration of how the reparameterization trick makes the $$\mathbf{z}$$ sampling process trainable.(Image source: Slide 12 in Kingma’s NIPS 2015 workshop [talk](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf))*

The reparameterization trick works for other types of distributions too, not only Gaussian.
In the multivariate Gaussian case, we make the model trainable by learning the mean and variance of the distribution, $$\mu$$ and $$\sigma$$, explicitly using the reparameterization trick, while the stochasticity remains in the random variable $$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$$.


![Gaussian VAE]({{ '/assets/images/vae-gaussian.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 9. Illustration of variational autoencoder model with the multivariate Gaussian assumption.*



## Beta-VAE

If each variable in the inferred latent representation $$\mathbf{z}$$ is only sensitive to one single generative factor and relatively invariant to other factors, we will say this representation is disentangled or factorized. One benefit that often comes with disentangled representation is *good interpretability* and easy generalization to a variety of tasks. 

For example, a model trained on photos of human faces might capture the gentle, skin color, hair color, hair length, emotion, whether wearing a pair of glasses and many other relatively independent factors in separate dimensions. Such a disentangled representation is very beneficial to facial image generation.

$$\beta$$-VAE ([Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl)) is a modification of Variational Autoencoder with a special emphasis to discover disentangled latent factors. Following the same incentive in VAE, we want to maximize the probability of generating real data, while keeping the distance between the real and estimated posterior distributions small (say, under a small constant $$\delta$$):


$$
\begin{aligned}
&\max_{\phi, \theta} \mathbb{E}_{\mathbf{x}\sim\mathcal{D}}[\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z})]\\
&\text{subject to } D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) < \delta
\end{aligned}
$$


We can rewrite it as a Lagrangian with a Lagrangian multiplier $$\beta$$ under the [KKT condition](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf). The above optimization problem with only one inequality constraint is equivalent to maximizing the following equation $$\mathcal{F}(\theta, \phi, \beta)$$:


$$
\begin{aligned}
\mathcal{F}(\theta, \phi, \beta) &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta(D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) - \delta) & \\
& = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) + \beta \delta & \\
& \geq \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) & \scriptstyle{\text{; Because }\beta,\delta\geq 0}
\end{aligned}
$$

The loss function of $$\beta$$-VAE is defined as:


$$
L_\text{BETA}(\phi, \beta) = - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z}))
$$

where the Lagrangian multiplier $$\beta$$ is considered as a hyperparameter.

Since the negation of $$L_\text{BETA}(\phi, \beta)$$ is the lower bound of the Lagrangian $$\mathcal{F}(\theta, \phi, \beta)$$. Minimizing the loss is equivalent to maximizing the Lagrangian and thus works for our initial optimization problem.

When $$\beta=1$$, it is same as VAE. When $$\beta > 1$$, it applies a stronger constraint on the latent bottleneck and limits the representation capacity of $$\mathbf{z}$$. For some conditionally independent generative factors, keeping them disentangled is the most efficient representation. Therefore a higher $$\beta$$ encourages more efficient latent encoding and further encourages the disentanglement. Meanwhile, a higher $$\beta$$ may create a trade-off between reconstruction quality and the extent of disentanglement. 

[Burgess, et al. (2017)](https://arxiv.org/pdf/1804.03599.pdf) discussed the distentangling in $$\beta$$-VAE in depth with an inspiration by the [information bottleneck theory]({{ site.baseurl }}{% post_url 2017-09-28-anatomize-deep-learning-with-information-theory %}) and further proposed a modification to $$\beta$$-VAE to better control the encoding representation capacity.


## VQ-VAE

Discrete VAE.

TBA.

## DRAW

Recurrent VAE.

TBA.


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at [lilian dot wengweng at gmail dot com] and I would be very happy to correct them right away!*

See you in the next post :D



## References

[1] Geoffrey E. Hinton, and Ruslan R. Salakhutdinov. ["Reducing the dimensionality of data with neural networks."](https://pdfs.semanticscholar.org/c50d/ca78e97e335d362d6b991ae0e1448914e9a3.pdf) Science 313.5786 (2006): 504-507.

[2] Pascal Vincent, et al. ["Extracting and composing robust features with denoising autoencoders."](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) ICML, 2008.

[3] Pascal Vincent, et al. ["Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion."](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf). Journal of machine learning research 11.Dec (2010): 3371-3408.

[4] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov. "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580 (2012).

[5] [Sparse Autoencoder](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) by Andrew Ng.

[6] Alireza Makhzani, Brendan Frey (2013). ["k-sparse autoencoder"](https://arxiv.org/abs/1312.5663). ICLR 2014.

[7] Salah Rifai, et al. ["Contractive auto-encoders: Explicit invariance during feature extraction."](http://www.icml-2011.org/papers/455_icmlpaper.pdf) ICML, 2011.

[8] Diederik P. Kingma, and Max Welling. ["Auto-encoding variational bayes.”](https://arxiv.org/abs/1312.6114) ICLR 2014.

[9] [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) on jaan.io

[10] Youtube tutorial: [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) by Arxiv Insights

[11] ["A Beginner's Guide to Variational Methods: Mean-Field Approximation"](https://blog.evjang.com/2016/08/variational-bayes.html) by Eric Jang.

[12] Carl Doersch. ["Tutorial on variational autoencoders."](https://arxiv.org/abs/1606.05908) arXiv:1606.05908, 2016.

[13] Irina Higgins, et al. ["$$\beta$$-VAE: Learning basic visual concepts with a constrained variational framework."](https://openreview.net/forum?id=Sy2fzU9gl) ICLR 2017.

[14] Christopher P. Burgess, et al. ["Understanding disentangling in beta-VAE."](https://arxiv.org/abs/1804.03599) NIPS 2017.




