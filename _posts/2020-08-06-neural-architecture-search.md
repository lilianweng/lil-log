---
layout: post
comments: true
title: "Neural Architecture Search"
date: 2020-08-06 12:00:00
tags: reinforcement-learning auto-ML
---


> Neural Architecture Search (NAS) automates network architecture engineering. By dissecting the NAS framework into three dimensions, the search space, the search algorithms and the child model evolution strategies, this post reviews many interesting ideas for better, faster and more cost-efficient automatic neural architecture search.


<!--more-->


Although most popular and successful model architectures are designed by human experts, automatically learning and evolving network topologies is not a new idea ([Stanley & Miikkulainen, 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)). In recent years, the pioneer work by [Zoph & Le 2017](https://arxiv.org/abs/1611.01578) and [Baker et al. 2017](https://arxiv.org/abs/1611.02167) has attracted a lot of attention into the field of Neural Architecture Search (NAS), leading to many interesting ideas for better, faster and more cost-efficient NAS methods.

As I started looking into the field of NAS, I found a nice survey on the topic by [Elsken, et al 2019](https://arxiv.org/abs/1808.05377) very helpful. They categorized NAS literature into the following framework with 3 components, which is clean, concise and also commonly adopted in most NAS literature.
**Search space**: The NAS search space defines a set of operations (e.g. convolution, fully-connected, pooling) and how operations can be connected into a valid network architecture. The design of search space usually involves human expertise, as well as human biases.
**Search algorithm**: A NAS search algorithm samples a population of network architecture candidates. It receives the child model performance as reward (e.g. high accuracy, low latency) and optimizes to generate high-performance architecture candidates.
**Evaluation strategy**: We need to evaluate, estimate or predict the performance of a large number of proposed child models in order to obtain feedback for the search algorithm so that we can learn how to construct better models. The process of candidate evaluation could be very expensive and many new evaluation methods have been proposed to save time or computation resources.


![High-level categorization of NAS]({{ '/assets/images/NAS-high-level.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 1. Three main components of Neural Architecture Search models.  (Image source: [Elsken, et al. 2019](https://arxiv.org/abs/1808.05377) with customized annotation in red)*


{: class="table-of-content"}
* TOC
{:toc}



## Search Space

The NAS search space defines a set of operations (e.g. convolution, fully-connected, pooling) and how operations can be connected into a valid network architecture. The design of search space usually involves human expertise, as well as human biases.


### Sequential layer-wise operations

The most naive way to design the search space for neural network architecture is the *sequential layer-wise operations*, meaning that every model, either CNN or RNN, is represented by a sequence of operations, as seen in early work of [Zoph & Le 2017](https://arxiv.org/abs/1611.01578) & [Baker et al. 2017](https://arxiv.org/abs/1611.02167). The serialization of model representation requires quite an amount of expert knowledge, as each operation is associated with different parameters and such association is hardcoded. For example, after predicting a `conv` op, the model should output kernel size, stride size, etc; or after predicting a `FC` op, we need to see the number of units as the next prediction. 


![The sequential layer-wise operation search space]({{ '/assets/images/NAS-search-space.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 2. (Top) A sequential representation of CNN. (Bottom) A sequential representation tree structure of a recurrent cell. (Image source: [Zoph & Le 2017](https://arxiv.org/abs/1611.01578))*

To make sure the generated architecture is valid, additional rules might be needed ([Zoph & Le 2017](https://arxiv.org/abs/1611.01578)):
- If a layer is not connected to any input layer then it is used as the input layer;
- Take all layer outputs that have not been connected and concatenate them at the final layer;
- If one layer has many input layers then all input layers are concatenated in the depth dimension;
- If input layers to be concatenated have different sizes, we pad the small layers with zeros so that the concatenated layers have the same sizes.

The skip connection can be predicted as well, using an attention-style mechanism. At layer $$N$$ , an anchor point is added which has $$N−1$$ content-based sigmoids to indicate which of the previous layers to be connected. Each sigmoid takes as input the hidden states of the current node $$h_i$$ and $$N-1$$ previous nodes $$h_j, j=1, \dots, N-1$$ .

$$
P(\text{Layer j is an input to layer i}) = \text{sigmoid}(v^\top \tanh(\mathbf{W}_\text{prev} h_j + \mathbf{W}_\text{curr} h_i))
$$

The sequential search space has a lot of representation power, but it is very large and can consume a ton of computation resources to fully cover the search space. In the experiments by [Zoph & Le 2017](https://arxiv.org/abs/1611.01578), they were running 800 GPUs in parallel for 28 days and [Baker et al. 2017](https://arxiv.org/abs/1611.02167) restricted the search space to contain at most 2 `FC` layers.



### Cell-based representation

Inspired by the design of using repeated modules in successful vision network architectures (e.g. Inception, ResNet), the *NASNet search space* ([Zoph et al. 2018](https://arxiv.org/abs/1707.07012)) defines the architecture of a conv net as the same cell getting repeated multiple times and each cell contains several operations predicted by the NAS algorithm. A well-designed cell module enables transferability between datasets.

Precisely, the NASNet search space learns two types of cells for network construction:
1. *Normal cell*: The input and output feature maps have the same dimension.
2. *Reduction cell*: The output feature map has its width and height reduced by half.


![NASNet search space]({{ '/assets/images/NASNet-search-space.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 3. The NASNet search space constrains the architecture as a repeated stack of cells. The cell architecture is optimized via NAS algorithms. (Image source: [Zoph et al. 2018](https://arxiv.org/abs/1707.07012))*

The predictions for each cell are grouped into $$B$$ blocks ($B=5$ in the paper), where each block has 5 prediction steps made by 5 distinct softmax classifiers corresponding to discrete choices of the elements of a block. Note that the NASNet search space does not have residual connections between cells and the model only learns skip connections on their own within blocks.


![5 prediction steps in one block]({{ '/assets/images/cell-prediction-steps.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 4. (a) Each cell consists of $$B$$ blocks and each block is predicted by 5 discrete decisions. (b)*


<a name="ScheduledDropPath"></a>During the experiments, they discovered that a modified version of [*DropPath*](https://arxiv.org/abs/1605.07648), named *ScheduledDropPath*, significantly improves the final performance of NASNet experiments. DropPath stochastically drops out paths (i.e. edges with operations attached in NASNet) with a fixed probability. ScheduledDropPath is DropPath with a linearly increasing probability of path dropping during training time.


[Elsken, et al 2019](https://arxiv.org/abs/1808.05377) pointed out three major advantages of the NASNet search space:
1. The search space size is reduced drastically;
2. The motif-based architecture can be more easily transferred to different datasets.
3. It demonstrates a strong proof of a useful design pattern in architecture engineering by repeated stacking modules; for example, we can build strong models by stacking residual blocks in CNN or stacking multi-headed attention blocks in Transformer.



### Hierarchical Structure

To take advantage of already discovered well-designed network [motifs](https://en.wikipedia.org/wiki/Network_motif), the NAS search space can be constrained as a hierarchical structure ([Liu et al 2017](https://arxiv.org/abs/1711.00436)), known as *Hierarchical NAS* (**HNAS**). It starts with a small set of primitives, including individual operations like convolution operation, pooling, identity, etc. Then small sub-graphs (or "motifs") that consistent of primitive operations are recursively used to form higher-level computation graphs.

A computation motif at level $$\ell=1, \dots, L$$ can be represented by $$(G^{(\ell)}, \mathcal{O}^{(\ell)})$$, where:
- $$\mathcal{O}^{(\ell)}$$ is a set of operations, $$\mathcal{O}^{(\ell)} = \{ o^{(\ell)}_1, o^{(\ell)}_2, \dots \}$$
- $$G^{(\ell)}$$ is an adjacency matrix, where the entry $$G_{ij}=k$$ indicates that operation $$o^{(\ell)}_k$$ is placed between node $$i$$ and $$j$$. The node indices follow topological ordering in DAG, where the index $1$ is the source and the maximal index is the sink node.



![Hierarchical search space]({{ '/assets/images/hierarchical-NAS-search-space.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 5. (Top) 3 level-1 primitive operations are composed into a level-2 motif. (Bottom) 3 level-2 motifs are plugged into a base network structure and assembled into a level-3 motif. (Image source: [Liu et al 2017](https://arxiv.org/abs/1711.00436))*


To construct the architecture according to the hierarchical structure, we start from the lowest level $$\ell=1$$ and recursively define the $$m$$-th motif operation at level $$\ell$$ as 

$$
o^{(\ell)}_m = \text{assemble}\Big( G_m^{(\ell)}, \mathcal{O}^{(\ell-1)} \Big)
$$ 


A hierarchical representation becomes $$\Big( \big\{ \{ G_m^{(\ell)} \}_{m=1}^{M_\ell} \big\}_{\ell=2}^L, \mathcal{O}^{(1)} \Big), \forall \ell=2, \dots, L$$, where $$\mathcal{O}^{(1)}$$ contains a set of primitive operations.

The $$\text{assemble}()$$ process is equivalent to sequentially compute the feature map of node $$i$$ by aggregating all the feature maps of its predecessor node $$j$$ following the topological ordering: 

$$
x_i = \text{merge} \big[ \{ o^{(\ell)}_{G^{(\ell)}_{ij}}(x_j) \}_{j < i} \big], i = 2, \dots, \vert G^{(\ell)} \vert
$$

where $$\text{merge}[]$$ is implemented as depth-wise concatenation in the paper.

Same as NASNet, experiments in [Liu et al 2017](https://arxiv.org/abs/1711.00436) focused on discovering good cell architecture within a predefined structure with repetitive modules. They showed that the power of simple search methods (e.g. random search or evolutionary algorithms) can be substantially enhanced using well-designed search spaces.

[Cai et al. 2018b](https://arxiv.org/abs/1806.02639) proposed a tree-structure search space using path-level network transformation. Each node in a tree structure defines an *allocation* scheme for splitting inputs for child nodes and a *merge* scheme for combining results from child nodes. The path-level network transformation allows replacing a single layer with a multi-branch motif if its corresponding merge scheme is add or concat.


![Path-level network transformation]({{ '/assets/images/path-level-network-transformations.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 6. An illustration of transforming a single layer to a tree-structured motif via path-level transformation operations. (Image source: [Cai et al. 2018b](https://arxiv.org/abs/1806.02639))*



### Memory-bank representation

A memory-bank representation of feed-forward networks is proposed by [Brock et al. 2017](https://arxiv.org/abs/1708.05344) in [SMASH](TBA). Instead of a graph of operations, they view a network as a system with multiple memory blocks which can read and write. Each layer operation is designed to: (1) read from a subset of memory blocks; (2) computes results; finally (3) write the results into another subset of blocks. For example, in a sequential model, a single memory block would get read and overwritten consistently.



![Memory-bank view]({{ '/assets/images/NAS-memory-bank-view-representation.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 7. Memory-bank representation of several popular network architecture blocks. (Image source: [Brock et al. 2017](https://arxiv.org/abs/1708.05344))*


## Search Algorithms

NAS search algorithms sample a population of network architecture candidates. It receives the child model performance as reward (e.g. high accuracy, low latency) and optimizes to generate high-performance architecture candidates. You may some relevancy to the field of hyperparameter search.


### Random search

Random search is to sample a valid architecture candidate from the search space at random and no learning model is involved. Random search has proved to be quite useful in hyperparameter search ([Bergstra & Bengio 2012](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)). With a well-designed search space, random search could be a quite challenging baseline.


### Reinforcement learning

The initial design of **NAS** ([Zoph & Le 2017](https://arxiv.org/abs/1611.01578)) involves a RL-based controller for proposing child model architectures for evaluation. The controller is implemented as a RNN, outputting a sequence of tokens used for configuring a network architecture. 


![NAS]({{ '/assets/images/NAS.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 8. A high level overview of NAS, containing a controller and a pipeline for evaluating child models. (Image source: [Zoph & Le 2017](https://arxiv.org/abs/1611.01578))*


The controller is trained as a *RL task* using [REINFORCE](TBA).
- **Action space**: The action space is a list of tokens for defining a child network predicted by the controller (See more in the above [section](TBA) on the sequential token search space). The controller outputs *action*, $$a_{1:T}$$, where $$T$$ is the total number of tokens.
- **Reward**: The accuracy of a child network that can be achieved at convergence is the reward for the controller model, $R$.
- **Loss**: NAS optimizes the controller parameters with a REINFORCE loss. We want to maximize the expected reward (high accuracy) with the gradient as follows. The nice thing here with policy gradient is that it works even when the reward is non-differentiable.

$$
\nabla_{\theta_c} J(\theta_c) = \sum_{t=1}^T \mathbb{E}[\nabla_{\theta_c} \log P(a_t \vert a_{1:(t-1)}; \theta_c) R ]
$$

**MetaQNN** ([Baker et al. 2017](https://arxiv.org/abs/1611.02167)) trains an agent to sequentially choose CNN layers using [*Q-learning*](TBA) with an [$$\epsilon$$-greedy](TBA) exploration strategy and experience replay. The reward is also the validation accuracy.

$$
Q^{(t+1)}(s_t,  a_t) = (1 - \alpha)Q^{(t)}(s_t, a_t) + \alpha (R_t + \gamma \max_{a \in \mathcal{A}} Q^{(t)}(s_{t+1},  a'))
$$

where a state $$s_t$$ is a tuple of layer operation and related parameters and an action $$a$ determines the connectivity between operations. The Q-value is proportional to how confident we are in two connected operations leading to high accuracy.


![MetaQNN]({{ '/assets/images/MetaQNN.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 9. Overview of MetaQNN - designing CNN models with Q-Learning. (Image source: [Baker et al. 2017](https://arxiv.org/abs/1611.02167))*



### Evolutionary algorithms

**NEAT** (short for *NeuroEvolution of Augmenting Topologies*; [Stanley & Miikkulainen, 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))is an approach for evolving neural network topology with [genetic algorithm (GA)](https://en.wikipedia.org/wiki/Genetic_algorithm), proposed in 2002. NEAT evolves both connection weights and network topologies together. Each gene encodes the full information of network architecture, including node weights and edges. The population grows by applying mutation of both weights and connections, as well as crossover between two parent genes.

![Mutation operations in NEAT]({{ '/assets/images/NEAT-mutations.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 10. Mutations in the NEAT algorithm. (Image source: Fig 3 & 4 in [Stanley & Miikkulainen, 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))*

For neuroevolution, please refer to this in-depth survey by [Stanley et al. 2019](https://www.nature.com/articles/s42256-018-0006-z) for more content.

[Real et al. (2018)](https://arxiv.org/abs/1802.01548) adopted the evolutionary algorithms (EA) as a way to search for high-performance network architecture, named **AmoebaNet**. They used the tournament selection method, which at each iteration picks the best candidate as parent among a set of samples (known as a tournament) drawn at random from the entire population and then places its mutated offspring back into the population. When the tournament size is $$1$$, it is equivalent to random selection.

When evolving the child model architectures, [Real et al. (2018)](https://arxiv.org/abs/1802.01548) modified the tournament selection to favor *younger* genotypes and discard the oldest models within each cycle. Such an approach,  named *aging evolution*, allows NAS to cover and explore more search space, rather than narrow down on good performance models too early. 

Precisely, in every iteration of the tournament selection with aging regularization (See Fig. X):
1. Sample $$S$$ models from the population and the one with highest accuracy is chosen as *parent*.
2. A *child* model is produced by mutating *parent*.
3. Then the child model is trained, evaluated and added back into the population.
4. The oldest model is removed from the population.


![Aging evolution]({{ '/assets/images/aging-evolution-NAS.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 11. The algorithm of aging evolution. (Image source: [Real et al. 2018](https://arxiv.org/abs/1802.01548))*


Two types of mutations are applied:
1. *Hidden state mutation*: randomly chooses a pairwise combination and re-wires a random end such that there is no loop in the graph.
2. *Op mutation*: randomly replaces an existing operation with a random one.


![2 mutations in AmoebaNet]({{ '/assets/images/AmoebaNet-mutations.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 12. Two types of mutations in AmoebaNet. (Image source: [Real et al. 2018](https://arxiv.org/abs/1802.01548))*

In their experiments, EA and RL works equally well in terms of the final validation accuracy, but EA has better anytime performance and is able to find smaller models. Here using EA in NAS is still expensive computation-wise, as each experiment took 7 days with 450 GPUs.

HNAS ([Liu et al 2017](https://arxiv.org/abs/1711.00436)) also used the evolutionary algorithms as their search strategy. In the hierarchical structure search space, each edge is one operation. Thus genotype mutation in their experiments is applied by replacing a random edge with a different operation. The replacement included the `none` op, so it can alter, remove and add an edge. The initial set of genotypes are created by applying a large number of random mutations on "trivial" motifs (all identity mappings). They use the original tournament selection as their search algorithm.



### Progressive Decision Process

Building up a model architecture is a sequential process. Every additional operator or layer adds on extra complexity. If we guide the search model to start investigation from simple models and gradually evolve to more complex architectures, it is like to introduce ["curriculum"](TBA) into the search model's learning process.

*Progressive NAS* (**PNAS**; [Liu, et al 2018](https://arxiv.org/abs/1712.00559)) frames the problem of architecture search as a progressive procedure for searching models of increasing complexity. Instead of RL or EA, PNAS adopts a sequential model-based bayesian optimization (SMBO) as search strategy. PNAS works similar to A* search, as it searches for models from simple to hard while simultaneously learning a surrogate function to guide the search.

> [A* search algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) ("best-first search") is a popular algorithm for path finding. The problem is framed as finding a path of smallest cost from a specific starting node to a given target node in a weighted graph. At each iteration, A* finds a path to extend by minimizing: $f(n)=g(n)+h(n)$, where $n$ is the next node, $g(n)$ is the cost from start to $n$, and $h(n)$ is the heuristic function that estimates the minimum cost of going from $n$ to the goal.
 
PNAS uses the same search space as [NASNet](TBA) and same as [Zoph et al. 2018](https://arxiv.org/abs/1707.07012) each block is specified as a 5-element tuple, but PNAS only considers the element-wise addition as the step 5 combination operator, no concatenation. However, instead of setting the number of blocks $$B$$ at a fixed number, PNAS starts with a model with only one block in a cell and gradually increases $$B=1, \dots, 5$$. 

The performance of these models on validaset set is used as rewards to learn a *surrogate* model for predicting the performance of novel architectures. With this performance predictor we can thus decide which models to evaluate next. Since the performance predictor should be able to handle various-sized inputs, accuracy, and sample-efficient, they ended up using a RNN model.
 

![Progressive NAS]({{ '/assets/images/progressive-NAS.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 12. The algorithm of Progressive NAS. (Image source: [Liu, et al 2018](https://arxiv.org/abs/1712.00559))*



### Gradient descent

Using gradient descent to update the architecture search model requires an effort to make the discrete operator choice differentiable. This approach usually combines learning both architecture parameters and network weights together into one model. See more in the (section)[TBA] on the *"one-shot"* approach.



## Evaluation Strategy

We need to evaluate, estimate or predict the performance of every child model in order to obtain feedback for the search algorithm. The process of candidate evaluation could be very expensive and many new evaluation methods have been proposed to save time or computation resources.

When evaluating a child model, we mostly care about its performance measured as accuracy on a validation set. Recent work starts looking into other factors of a model, such as model size and latency, as certain devices may have limitations on memory or demand fast response time.


### Training from Scratch

The most naive approach is to train a child network from scratch on a training dataset until *convergence* and then measure its accuracy on a validation set ([Zoph & Le 2017](https://arxiv.org/abs/1611.01578)). It gives us solid performance numbers, but one full training-convergence-evaluation loop only provides one data sample for training the RL controller. Thus it is very expensive in terms of computation consumption.


### Proxy Task Performance

There are several approaches for using a proxy task performance as the performance estimation of a child network, which is generally cheaper and faster to measure:
- Train on a smaller dataset.
- Train for a fewer epochs.
- Train and evaluate a down-scaled model in the search stage. For example, once a cell structure is learned, we can play with the number of cell repeats or scale up the number of  filters ([Zoph et al. 2018](https://arxiv.org/abs/1707.07012)).
- Predict the learning curve. [Baker et al (2018)](https://arxiv.org/abs/1705.10823) models the validation accuracies as a time-series regression problem. The features for the regression model ($$\nu$$-support vector machine regressions; $$\nu$$-SVR) contains the accuracies, architecture parameters and hyperparameters.


### Parameter Sharing

Instead of training every child model independently from scratch. You may ask, ok, what if we construct dependency between them and find a way to reuse weights? Some researchers actually succeeded to make such approaches work.

Inspired by [Net2net](https://arxiv.org/abs/1511.05641) transformation, [Cai et al (2017)](https://arxiv.org/abs/1707.04873) proposed *Efficient Architecture Search* (**EAS**). EAS has a RL agent as a meta-controller to predict function-preserving network transformations so as to grow the network depth or layer width. Because the network is growing incrementally, the weights of previously validated networks can be *reused* for further exploration. With inherited weights, newly constructed networks only need some light-weighted training.

A meta-controller learns to generate *network transformation actions* given the current network architecture, which is specified with a variable-length string. 
- In order to handle architecture configuration of a variable length, the meta-controller is implemented as a bi-directional recurrent network.
- Multiple actor networks output different output different transformation decisions, such as
    1. *Net2WiderNet* operation allows to replace a layer with a wider layer, meaning more units for fully-connected layers, or more filters for convolutional layers, while preserving the functionality.
    2. *Net2DeeperNet* operation allows to insert a new layer that is initialized as adding an identity mapping between two layers so as to preserve the functionality.


![EAS meta-controller]({{ '/assets/images/EAS-meta-controller.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 13. Overview of the RL based meta-controller in Efficient Architecture Search (NAS). After encoding the architecture configuration, it outputs net2net transformation actions through two separate actor networks. (Image source: [Cai et al 2017](https://arxiv.org/abs/1707.04873))*
 

With similar motivation, *Efficient NAS* (**ENAS**; [Pham et al. 2018](https://arxiv.org/abs/1802.03268)) speeds up NAS (i.e. 1000x less) by aggressively sharing parameters among child models. The core idea of ENAS is the observation that all of the sampled architecture graphs can be viewed as *sub-graphs* of a larger *supergraph*. All the child networks are sharing weights of this supergraph.


![ENAS example]({{ '/assets/images/ENAS-example.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 14. (Left) The graph represents the entire search space for a 4-node recurrent cell, but only connections in red are active. (Middle) An example of the left active sub-graph can be translated into a child model architecture. (Right) The network config parameters output by a RNN controller for the architecture in the middle. (Image source: [Pham et al. 2018](https://arxiv.org/abs/1802.03268))*


ENAS alternates between training the shared model weights $$\omega$$ and training the controller $$\theta$$:
1. The parameters of the controller LSTM $$\theta$$ are trained with REINFORCE, where the reward $$\mathcal{R}(\mathbf{m}, \omega)$$ is computed on the validation set.
2. The shared parameters of the child models $$\omega$$ are trained with standard supervised learning loss. Note that different operators associated with the same node in the supergraph would have their own distinct parameters.


### Prediction-Based

A routine child model evaluation loop is to update model weights through gradient descent over the training and validation datasets. SMASH ([Brock et al. 2017](https://arxiv.org/abs/1708.05344)) proposed a different and interesting idea: how about we predict the model weights directly based on the network architecture parameters?

They employed a [HyperNet](https://blog.otoro.net/2016/09/28/hyper-networks/) ([Ha et al 2016](https://arxiv.org/abs/1609.09106)) to generated the weights of a model conditioned on an encoding of its architecture. Then the model with HyperNet-generated weights can be validated and provide performance reward. Note that we don't need extra training for every child model but we do need to train the HyperNet.


![SMASH algorithm]({{ '/assets/images/SMASH-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 15. The algorithm of SMASH. (Image source: [Brock et al. 2017](https://arxiv.org/abs/1708.05344))*

A correlation between the model performance with SMASH-generated weights and true validation errors suggests that predicted weights can be used for model comparison, to some extent. We do need a HyperNet of large enough capacity, as the correlation is corrupted if the HyperNet model is too small compared to the child model size.


![SMASH error correlation]({{ '/assets/images/SMASH-error-correlation.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 16. The algorithm of SMASH. (Image source: [Brock et al. 2017](https://arxiv.org/abs/1708.05344))*


SMASH can be viewed as another way to implement child model parameter sharing. One problem as pointed in [Pham et al. 2018](https://arxiv.org/abs/1802.03268) is: the usage of HyperNet restricts the weights of SMASH child models to a low-rank space, because weights are generated via tensor products. In comparison, ENAS has no such restrictions.



## One-Shot Approach: Search + Evaluation

Doing search & evaluation independently for a large population of child models is expensive. We have seen promising approaches like [Brock et al. 2017](https://arxiv.org/abs/1708.05344) or [Pham et al. 2018](https://arxiv.org/abs/1802.03268), where training a single model is enough for emulating any child model in the search space. 

The **one-shot** architecture search follows the same idea of weight sharing and furthermore it combines the learning of architecture generation together with parameter weights. The following approaches all treat every child architecture as different sub-graphs of a supergraph and share weights between sub-graphs with edges in common.

[Bender et al (2018)](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf) constructed a single large over-parameterized network, known as the *One-Shot model*, such that it contains every possible operation in the search space. With [ScheduledDropPath](#ScheduledDropPath) (the dropout rate is increased over time, which is $$r^{1/k}$$ at the end of training, where $$0 < r < 1$$ is a hyperparam and $$k$$ is the number of incoming paths) and some carefully designed tricks (e.g. ghost batch normalization, L2 regularization only on the active architecture), the training of such a giant model can be stabilized enough and then used for evaluating any child model sampled from the supergraph.


![One-Shot model architecture]({{ '/assets/images/one-shot-model-architecture.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 17. The architecture of the one-shot model in [Bender et al 2018](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf). Each cell has $$N$$ choice blocks and each choice block can select up to 2 operations. Solid edges are used in every architecture, where dash lines are optional. (Image source: [Bender et al 2018](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf))*


Once the one-shot model is trained, it can be used for evaluating the performance of many different architectures sampled at random by zeroing out or removing some operations. This sampling process can be replaced by RL or evolution.

They observed that the accuracy of a sample architecture in the one-shot model versus the same architecture after a small fine-tuning could be very large. The hypothesis is that the one-shot model automatically learns to focus on the *most useful* operations in the network and comes to *rely on* these operations when they are available. Thus removing less important components only causes a small impact, but zeroing out useful operations lead to big reduction in model accuracy --- therefore, we see a larger variance in scores when using the one-shot model for evaluation.


![One-shot accuracy]({{ '/assets/images/one-shot-model-accuracy-correlation.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 18. A stratified sample of models with different one-shot model accuracy versus their true validation accuracy as stand-alone models. (Image source: [Bender et al 2018](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf))*


Clearly designing such a search graph is not a trivial task, but it demonstrates that one shot approach works well with only gradient descent and no additional algorithm like RL or EA is a must.

[Liu et al (2019)](https://arxiv.org/abs/1806.09055) believes that one main cause for inefficiency in NAS is to treat the architecture search as a black-box optimization and thus we fall into methods like RL, evolution, SMBO, etc. If we can rely on standard gradient descent, we could potentially optimize the search process more effectively. As a result, they proposed *Differentiable Architecture Search* (**DARTS**; [Liu et al 2019](https://arxiv.org/abs/1806.09055)), which introduces a continuous relaxation on each path in the search graph, making it possible to jointly train architecture parameters and weights via gradient descent.

Here let's use the directed acyclic graph (DAG) representation introduced [earlier](TBA). A cell is a DAG consisting of a topologically ordered sequence of $$N$$ nodes. Each node has a latent representation $$x_i$$ to be learned. Each edge $$(i, j)$$ is tied to some operation $$o^{(i,j)} \in \mathcal{O}$$ that transforms $$x_j$$ to construct $$x_i$$:

$$
x_i = \sum_{j < i} o^{(i,j)}(x_j)
$$

To make the search space continuous, DARTS relaxes the categorical choice of a particular operation as a softmax over all the operations and the task of architecture search is reduced to learn a set of mixing probabilities $$\alpha = \{ \alpha^{(i,j)} \}$$. 


$$
\bar{o}^{(i,j)}(x) = \sum_{o\in\mathcal{O}} \frac{\exp(\alpha_{ij}^o)}{\sum_{o'\in\mathcal{O}} \exp(\alpha^{o'}_{ij})} o(x)
$$

where $$\alpha_{ij}$$ is a vector of dimension $$\vert \mathcal{O} \vert$$, containing weights between nodes $$i$$ and $$j$$ over different operations. 

The bilevel optimization exists as we want to optimize both the network weights $$w$$ (for evaluation) and the architecture representation $$\alpha$$.

$$
\begin{aligned}
\min_\alpha & \mathcal{L}_\text{validate} (w^*(\alpha), \alpha) \\
\text{s.t.} & w^*(\alpha) = \arg\min_w \mathcal{L}_\text{train} (w, \alpha)
\end{aligned}
$$

At step $$k$$, given the current architecture parameters $$\alpha_{k−1$}$, we can optimize weights  $$w_k$$ by moving $$w_{k−1}$$ in the direction of minimizing the training loss $$\mathcal{L}_\text{train}(w_{k−1}, \alpha_{k−1})$$ with a learning rate $$\xi$$. Next, while keeping the newly updated weights $$w_k$$ fixed, we update the mixing probabilities so as to minimize the validation loss *after a single step of gradient descent w.r.t. the weights*:


$$
J_\alpha = \mathcal{L}_\text{val}(w_k - \xi \nabla_w \mathcal{L}_\text{train}(w_k, \alpha_{k-1}), \alpha_{k-1})
$$

The motivation here is that we want to find an architecture with a low validation loss when its weights are optimized by gradient descent and the one-step unrolled weights serve as the *surrogate* for $$w^∗(\alpha)$$. 

> Side note: Earlier we have seen similar formulation in [MAML](TBA) where the two-step optimization happens between task losses and the meta-learner update, as well as framing [Domain Randomization](TBA) as a bilevel optimization for better transfer in the real environment.


![DARTS]({{ '/assets/images/DARTS-illustration.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 19. An illustration of how DARTS applies continuous relaxation on edges in DAG supergraph and identifies the final model. (Image source: [Liu et al 2019](https://arxiv.org/abs/1806.09055))*


$$
\begin{aligned}
\text{Let }w'_k &= w_k - \xi \nabla_w \mathcal{L}_\text{train}(w_k, \alpha_{k-1}) & \\
J_\alpha &= \mathcal{L}_\text{val}(w_k - \xi \nabla_w \mathcal{L}_\text{train}(w_k, \alpha_{k-1}), \alpha_{k-1}) = \mathcal{L}_\text{val}(w'_k, \alpha_{k-1}) & \\
\nabla_\alpha J_\alpha 
&= \nabla_{\alpha_{k-1}} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1}) \nabla_\alpha \alpha_{k-1} + \nabla_{w'_k} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1})\nabla_\alpha w'_k & \text{; multivariable chain rule}\\
&= \nabla_{\alpha_{k-1}} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1}) + \nabla_{w'_k} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1}) \big( - \xi \color{red}{\nabla^2_{\alpha, w} \mathcal{L}_\text{train}(w_k, \alpha_{k-1})} \big) & \\
&\approx \nabla_{\alpha_{k-1}} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1}) - \xi \nabla_{w'_k} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1}) \color{red}{\frac{\nabla_\alpha \mathcal{L}_\text{train}(w_k^+, \alpha_{k-1}) - \nabla_\alpha \mathcal{L}_\text{train}(w_k^-, \alpha_{k-1}) }{2\epsilon}} & \text{; numerical differentiation approximation}
\end{aligned}
$$

where the red part is using numerical differentiation approximation where $$w_k^+ = w_k + \epsilon \nabla_{w'_k} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1})$$ and $$w_k^- = w_k - \epsilon \nabla_{w'_k} \mathcal{L}_\text{val}(w'_k, \alpha_{k-1})$$.


![DARTS algorithm]({{ '/assets/images/DARTS-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 20. The algorithm overview of DARTS. (Image source: [Liu et al 2019](https://arxiv.org/abs/1806.09055))*


As another idea similar to DART, Stochastic NAS ([Xie et al., 2019](https://arxiv.org/abs/1812.09926)) applies a continuous relaxation by employing the concrete distribution (CONCRETE = CONtinuous relaxations of disCRETE random variables; [Maddison et al 2017](https://arxiv.org/abs/1611.00712)) and reparametrization tricks. The goal is the same as DARTS, to make the discrete distribution differentiable and thus enable optimization by gradient descent. 
<!--- TBA: maybe add more details on SNAS -->


DARTS is able to greatly reduce the cost of GPU hours. Their experiments for searching for CNN cells have $$N=7$$ and only took 1.5 days with a single GPU. However, it suffers from the high GPU memory consumption issue due to its continuous representation of network architecture. In order to fit the model into the memory of a single GPU, they picked a small number $$N$$.


To constrain the GPU memory consumption, **ProxylessNAS** ([Cai et al., 2019](https://arxiv.org/abs/1812.00332)) views NAS as a path-level pruning process in DAG and binarizes the architecture parameters to force only one path to be active between two nodes at a time. The probabilities for an edge being either masked out or not are then learned by sampling a few binarized architectures and using *BinaryConnect* ([Courbariaux et al., 2015](TBA)) to update the corresponding probabilities. ProxylessNAS demonstrates a strong connection between NAS and model compression. By using path-level compression, it is able to save memory consumption by one order of magnitude.

Let's continue with the graph representation. in a DAG adjacency matrix $$G$$ where $$G_{ij}$$ represents an edge between node $i$ and $j$ and its value can be chosen from the set of $$\vert \mathcal{O} \vert$$ candidate primitive operations, $\mathcal{O} = \{ o_1, \dots \}$. The One-Shot model, DARTS and ProxylessNAS all consider each edge as a mixture of operations, $$m_\mathcal{O}$$ but with different tweaks. 

In One-Shot, $$m_\mathcal{O}(x)$$ is the sum of all the operations. In DARTS, it is a weighted sum where weights are softmax over a real-valued architecture weighting vector $$\alpha$$ of length $$\vert \mathcal{O} \vert$$. ProxylessNAS transforms the softmax probabilities of $$\alpha$$ into a binary gate and uses the binary gate to keep only one operation active at a time.


$$
\begin{aligned}
m^\text{one-shot}_\mathcal{O}(x) &= \sum_{i=1}^{\vert \mathcal{O} \vert} o_i(x) \\
m^\text{DARTS}_\mathcal{O}(x) &= \sum_{i=1}^{\vert \mathcal{O} \vert} p_i o_i(x) = \sum_{i=1}^{\vert \mathcal{O} \vert} \frac{\exp(\alpha_i)}{\sum_j \exp(\alpha_j)} o_i(x) \\
m^\text{binary}_\mathcal{O}(x) &= \sum_{i=1}^{\vert \mathcal{O} \vert} g_i o_i(x) = \begin{cases}
o_1(x) & \text{with probability }p_1, \\
\dots &\\
o_{\vert \mathcal{O} \vert}(x) & \text{with probability }p_{\vert \mathcal{O} \vert}
\end{cases} \\
\text{ where } g &= \text{binarize}(p_1, \dots, p_N) = \begin{cases}
[1, 0, \dots, 0] & \text{with probability }p_1, \\
\dots & \\
[0, 0, \dots, 1] & \text{with probability }p_N. \\
\end{cases}
\end{aligned}
$$


![Training steps of ProxylessNAS]({{ '/assets/images/proxylessNAS-training.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 21. ProxylessNAS has two training steps running alternatively. (Image source: [Cai et al., 2019](https://arxiv.org/abs/1812.00332))*


ProxylessNAS runs two training steps alternatively:
1. When training weight parameters $$w$$, it freezes the architecture parameters $$\alpha$$ and stochastically samples binary gates according to the above $$m^\text{binary}_\mathcal{O}(x)$$. The weight parameters can be updated with standard gradient descent.
2. When training architecture parameters $$\alpha$$, it freezes $$w$$, resets the binary gates and then updates $$\alpha$$ on the validation set. Following the idea of *BinaryConnect*,  the gradient w.r.t. architecture parameters can be approximately estimated using $$\partial L / \partial g_i$$ in replacement for $$\partial L / \partial p_i$$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \alpha_i} 
&= \sum_{j=1}^{\vert \mathcal{O} \vert} \frac{\partial \mathcal{L}}{\partial p_j} \frac{\partial p_j}{\partial \alpha_i} 
\approx \sum_{j=1}^{\vert \mathcal{O} \vert} \frac{\partial \mathcal{L}}{\partial g_j} \frac{\partial p_j}{\partial \alpha_i} 
= \sum_{j=1}^{\vert \mathcal{O} \vert} \frac{\partial \mathcal{L}}{\partial g_j} \frac{\partial \frac{e^{\alpha_j}}{\sum_k e^{\alpha_k}}}{\partial \alpha_i} \\
&= \sum_{j=1}^{\vert \mathcal{O} \vert} \frac{\partial \mathcal{L}}{\partial g_j} \frac{\sum_k e^{\alpha_k} (\mathbf{1}_{i=j} e^{\alpha_j}) - e^{\alpha_j} e^{\alpha_i} }{(\sum_k e^{\alpha_k})^2}
= \sum_{j=1}^{\vert \mathcal{O} \vert} \frac{\partial \mathcal{L}}{\partial g_j} p_j (\mathbf{1}_{i=j} -p_i)
\end{aligned}
$$

Instead of BinaryConnect, REINFORCE can also be used for parameter updates with the goal for maximizing the reward, while no RNN meta-controller is involved.

Computing $$\partial L / \partial g_i$$ needs to calculate and store $$o_i(x)$$, which requires $$\vert \mathcal{O} \vert$$ times GPU memory. To resolve this issue, they factorize the task of choosing one path out of $N$ into multiple binary selection tasks (Intuition: "if a path is the best choice, it should be better than any other path"). At every update step, only 2 paths are sampled while others are masked. These 2 selected paths are updated according to the above equation and then scaled properly so that other path weights are unchanged. After this process, one of the sampled paths is enhanced (path weight increases) and the other is attenuated (path weight decreases) while all other paths keep unchanged.

Besides accuracy, ProxylessNAS also considers latency as an important metric to optimize, as different devices might have very different requirements on inference time latency (e.g. GPU, CPU, mobile). To make latency differentiable, they model latency as a continuous function of the network dimensions. The expected latency of a mixed operation can be written as $$\mathbb{E}[\text{latency}] = \sum_j p_j F(o_j)$$, where $F(.)$ is a latency prediction model:


![proxylessNAS latency]({{ '/assets/images/proxylessNAS-latency.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 22. Add a differentiable latency loss into the training of ProxylessNAS.  (Image source: [Cai et al., 2019](https://arxiv.org/abs/1812.00332))*



## What's the Future?

We have seen many interesting new ideas on automating the network architecture engineering through neural architecture search and they achieved very impressive performance. However it is a bit hard to do inference on *why* some architecture works and how we can develop modules generalizable across tasks rather than being very dataset-specific.

As also noted in [Elsken, et al 2019](https://arxiv.org/abs/1808.05377): "..., so far it provides little insights into why specific architectures work well and how similar the architectures derived in independent runs would be. Identifying common motifs, providing an understanding why those motifs are important for high performance, and investigating if these motifs generalize over different problems would be desirable."

In the meantime, purely focusing on improvement over validation accuracy might not be enough ([Cai et al., 2019](https://arxiv.org/abs/1812.00332)). Devices like mobiles for daily usage have limited memory and computation power. While AI applications are on the way to affect our daily life, it is unavoidable to be more *device-specific*.

Another interesting investigation is to consider *unlabelled dataset* and [self-supervised learning](TBA) for NAS. The size of labelled dataset is always limited and it is difficult to tell whether such a dataset has biases or big deviation from the real world data distribution. 

[Liu et al (2020)](TBA) looked into the question *"Can we find high-quality neural architecture without human-annotated labels?"* and proposed a new setup called *Unsupervised Neural Architecture Search* (**UnNAS**). The quality of the architecture needs to be estimated in an unsupervised fashion during the search phase. The paper experimented with three unsupervised [pretext tasks](TBA): image rotation prediction, colorization, and solving the jigsaw puzzle. 

They observed that:
1. High rank correlation between supervised accuracy and pretext accuracy *on the same dataset*. Typically the rank correlation is higher than 0.8, regardless of the dataset, the search space, and the pretext task.
2. High rank correlation between supervised accuracy and pretext accuracy *across datasets*.
3. Better pretext accuracy translates to better supervised accuracy.
4. Performance of UnNAS architecture is comparable to supervised counterparts, though not better yet.

They hypothesize that the architecture quality is correlated with image statistics. As CIFAR-10 and ImageNet are all on the natural images, they are comparable and the results are transferable. UnNAS could potentially enable a much larger amount of unlabelled data which captures image statistics better into the search phase.


Hyperparameter search is a long-standing topic in the ML community. And NAS automates architecture engineering. We are trying to automate processes in ML which used to configured by humans. Taking even one more step further, is it possible to automatically discover ML algorithms? **AutoML-Zero** ([Real et al 2020](https://arxiv.org/abs/2003.03384)) investigated this idea. Using [aging evolutionary algorithms](TBA), AutoML-Zero automatically searches for whole ML algorithms using little restriction on the form with only simple mathematical operations as building blocks.

It learns three component functions. Each function only adopts very basic operations.
- `Setup`: Initialize memory variables (weights).
- `Learn`: Modify memory variables
- `Predict`: Make a prediction from an input $$x$$.
 

![AutoML-zero evaluation]({{ '/assets/images/AutoML-zero-evaluation.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 23. Algorithm evaluation on one task (Image source: [Real et al 2020](https://arxiv.org/abs/2003.03384))*
 
Three types of mutations could be applied to a parent genotype:
1. Insert a random instruction or remove an instruction at a random location in a component function;
2. Randomize all the instructions in a component function;
3. Modify one of the arguments of an instruction by replacing it with a random choice (e.g. "swap the output address" or "change the value of a constant")


![Progress of AutoML-zero experiment]({{ '/assets/images/AutoML-zero-progress.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Figure 24. An illustration of evolutionary progress on projected binary CIFAR-10 with example code. (Image source: [Real et al 2020](https://arxiv.org/abs/2003.03384))*


## Appendix: Summary of NAS Papers

<table class="info">
    <thead>
        <tr>
            <th>Model name</th>
            <th>Search space</th>
            <th>Search algorithms</th>
            <th>Child model evaluation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>NEAT ([Stanley & Miikkulainen, 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))</td>
            <td>-</td>
            <td>Evolution (Genetic algorithm)</td>
            <td>-</td>
        </tr>
        <tr>
            <td>NAS ([Zoph & Le 2017](https://arxiv.org/abs/1611.01578))</td>
            <td>Sequential layer-wise ops</td>
            <td>RL (REINFORCE)</td>
            <td>Train from scratch until convergence</td>
        </tr>
        <tr>
            <td>MetaQNN ([Baker et al. 2017](https://arxiv.org/abs/1611.02167))</td>
            <td>Sequential layer-wise ops</td>
            <td>RL (Q-learning with $\epsilon$-greedy)</td>
            <td>Train for 20 epochs</td>
        </tr>
        <tr>
            <td>NASNet ([Zoph et al. 2018](https://arxiv.org/abs/1707.07012))</td>
            <td>Cell-based</td>
            <td>RL (PPO)</td>
            <td>Train for 20 epochs</td>
        </tr>
        <tr>
            <td>AmoebaNet ([Real et al 2018](https://arxiv.org/abs/1802.01548))</td>
            <td>NASNet search space</td>
            <td>Evolution (Tournament selection)</td>
            <td>Train for 25 epochs</td>
        </tr>
        <tr>
            <td>HNAS ([Liu et al 2017](https://arxiv.org/abs/1711.00436))</td>
            <td>Hierarchical structure</td>
            <td>Evolution (Tournament selection)</td>
            <td>Train for a fixed number of iterations</td>
        </tr>
        <tr>
            <td>EAS ([Cai et al 2018a](https://arxiv.org/abs/1707.04873))</td>
            <td>Network transformation</td>
            <td>RL (REINFORCE)</td>
            <td>2-stage training</td>
        </tr>
        <tr>
            <td>PNAS ([Liu et al. 2018](https://arxiv.org/abs/1712.00559))</td>
            <td>Reduced version of NASNet search space</td>
            <td>SMBO; Progressive search for architectures of increasing complexity</td>
            <td>Train for 20 epochs</td>
        </tr>
        <tr>
            <td>ENAS ([Pham et al. 2018](https://arxiv.org/abs/1802.03268))</td>
            <td>Both sequential and cell-based search space</td>
            <td>RL (REINFORCE)</td>
            <td>Train one model with shared weights</td>
        </tr>
        <tr>
            <td>SMASH ([Brock et al. 2017](https://arxiv.org/abs/1708.05344))</td>
            <td>Memory-bank representation</td>
            <td>Random search</td>
            <td>HyperNet predicts weights of evaluated architectures.</td>
        </tr>
        <tr>
            <td>One-Shot ([Bender et al 2018](http://proceedings.mlr.press/v80/bender18a.html))</td>
            <td>An over-parameterized one-shot model</td>
            <td>Random search (zero out some paths at random)</td>
            <td>Train the one-shot model</td>
        </tr>
        <tr>
            <td>DARTS ([Liu et al 2019](https://arxiv.org/abs/1806.09055))</td>
            <td>NASNet search space</td>
            <td colspan="2">Gradient descent (Softmax weights over operations)</td>
        </tr>
        <tr>
            <td>ProxylessNAS ([Cai et al 2019](https://arxiv.org/abs/1812.00332))</td>
            <td>Tree structure architecture</td>
            <td colspan="2">Gradient descent (BinaryConnect) or REINFORCE</td>
        </tr>
        <tr>
            <td>Stochastic NAS ([Xie et al 2019](https://arxiv.org/abs/1812.09926))</td>
            <td>NASNet search space</td>
            <td colspan="2">Gradient descent (concrete distribution)</td>
    </tbody>
</table>


## Reference

[1] Thomas Elsken, Jan Hendrik Metzen, Frank Hutter. ["Neural Architecture Search: A Survey"](https://arxiv.org/abs/1808.05377) JMLR 20 (2019) 1-21.

[2] Kenneth O. Stanley, et al. ["Designing neural networks through neuroevolution"](https://www.nature.com/articles/s42256-018-0006-z) Nature Machine Intelligence volume 1, pages 24–35 (2019).

[3] Kenneth O. Stanley & Risto Miikkulainen. ["Evolving Neural Networks through Augmenting Topologies"](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) Evolutionary Computation 10(2): 99-127 (2002).

[4] Barret Zoph, Quoc V. Le. ["Neural architecture search with reinforcement learning"](https://arxiv.org/abs/1611.01578) ICLR 2017.

[5] Bowen Baker, et al. ["Designing Neural Network Architectures using Reinforcement Learning"](https://arxiv.org/abs/1611.02167) ICLR 2017.

[6] Bowen Baker, et al. ["Accelerating neural architecture search using performance prediction"](https://arxiv.org/abs/1705.10823) ICLR Workshop 2018.

[7] Barret Zoph, et al. ["Learning transferable architectures for scalable image recognition"](https://arxiv.org/abs/1707.07012) CVPR 2018.

[8] Hanxiao Liu, et al. ["Hierarchical representations for efficient architecture search."](https://arxiv.org/abs/1711.00436) ICLR 2018.

[9] Esteban Real, et al. ["Regularized Evolution for Image Classifier Architecture Search"](https://arxiv.org/abs/1802.01548) arXiv:1802.01548 (2018).

[10] Han Cai, et al. ["Efficient architecture search by network transformation"] AAAI 2018a.

[11] Han Cai, et al. ["Path-Level Network Transformation for Efficient Architecture Search"](https://arxiv.org/abs/1806.02639) ICML 2018b.

[12] Han Cai, Ligeng Zhu & Song Han. ["ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware"](https://arxiv.org/abs/1812.00332) ICLR 2019.

[13] Chenxi Liu, et al. ["Progressive neural architecture search"](https://arxiv.org/abs/1712.00559) ECCV 2018.

[14] Hieu Pham, et al. ["Efficient neural architecture search via parameter sharing"](https://arxiv.org/abs/1802.03268) ICML 2018.

[15] Andrew Brock, et al. ["SMASH: One-shot model architecture search through hypernetworks."](https://arxiv.org/abs/1708.05344) ICLR 2018.

[16] Gabriel Bender, et al. ["Understanding and simplifying one-shot architecture search."](http://proceedings.mlr.press/v80/bender18a.html) ICML 2018.

[17] Hanxiao Liu, Karen Simonyan, Yiming Yang. ["DARTS: Differentiable Architecture Search"](https://arxiv.org/abs/1806.09055) ICLR 2019.

[18] Sirui Xie, Hehui Zheng, Chunxiao Liu, Liang Lin. ["SNAS: Stochastic Neural Architecture Search"](https://arxiv.org/abs/1812.09926) ICLR 2019.

[19] Chenxi Liu et al. ["Are Labels Necessary for Neural Architecture Search?"](https://arxiv.org/abs/2003.12056) ECCV 2020.

[20] Esteban Real, et al. ["AutoML-Zero: Evolving Machine Learning Algorithms From Scratch"](https://arxiv.org/abs/2003.03384) ICML 2020.

