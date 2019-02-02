---
layout: post
comments: true
title: "Generalized Language Models"
date: 2019-01-31 12:00:00
tags: nlp
image: "elmo-and-bert.png"
---

> As a follow up of word embedding post, we will discuss the models on learning contextualized word vectors, as well as the new trend in large unsupervised pre-trained language models which have achieved amazing SOTA results on a variety of language tasks.

<!--more-->

<br />
![Elmo & Bert]({{ '/assets/images/elmo-and-bert.png' | relative_url }})
{: style="width: 60%;" class="center"}
*Fig. 0. I guess they are Elmo & Bert? (Image source: [here](http://www.essentialkids.com.au/entertainment/tv/elmo-big-bird-lend-a-hand-20110825-1jcqt))*
<br />

We have seen amazing progress in NLP in 2018. Large-scale pre-trained language modes like [OpenAI GPT](https://blog.openai.com/language-unsupervised/) and [BERT](https://arxiv.org/abs/1810.04805) have achieved great performance on a variety of language tasks using generic model architectures. The idea is similar to how ImageNet classification pre-training helps many vision tasks (\*). Even better than vision classification pre-training, this simple and powerful approach in NLP does not require labeled data for pre-training, allowing us to experiment with increased training scale, up to our very limit.

*(\*) Although recently He et al. (2018) [found](https://arxiv.org/abs/1811.08883) that pre-training might not be necessary for image segmentation task.*

In my previous NLP [post on word embedding]({{ site.baseurl }}{% post_url 2017-10-15-learning-word-embedding %}), the introduced embeddings are not context-specific --- they are learned based on word concurrency but not sequential context. So in two sentences, "*I am eating an apple*" and "*I have an Apple phone*", two "apple" words refer to very different things but they would still share the same word embedding vector. 

Despite this, early adoption of word embeddings in problem-solving is to use them as additional features for an existing task-specific model and in a way the improvement is bounded.

In this post, we will discuss how various approaches were proposed to make embeddings dependent on context, and to make them easier and cheaper to be applied to downstream tasks in general form.


{: class="table-of-content"}
* TOC
{:toc}


## Metric: Perplexity

[Perplexity](https://en.wikipedia.org/wiki/Perplexity) is a simple metric widely used to gauge how well a language model can capture the real word distribution conditioned on the context. It is the negative log-likelihood predicted for the actual words. The smaller the better.


## CoVe

**CoVe** ([McCann et al. 2017](https://arxiv.org/abs/1708.00107)), short for **Contextual Word Vectors**, is a type of word embeddings learned by an encoder in an [attentional seq-to-seq]({{ site.baseurl }}{% post_url 2018-06-24-attention-attention %}#born-for-translation) machine translation model.
Different from traditional word embeddings introduced [here]({{ site.baseurl }}{% post_url 2017-10-15-learning-word-embedding %}), CoVe word representations are functions of the entire input sentence.


### NMT Recap

Here the Neural Machine Translation ([NMT](https://github.com/THUNLP-MT/MT-Reading-List)) model is composed of a standard, two-layer, bidirectional LSTM encoder and an attentional two-layer unidirectional LSTM decoder. It is pre-trained on the English-German translation task. The encoder learns and optimizes the embedding vectors of English words in order to translate them to German. With the intuition that the encoder should capture high-level semantic and syntactic meanings before transforming words into another language, the encoder output is used to provide contextualized word embeddings for various downstream language tasks.


![NMT Recap]({{ '/assets/images/nmt-recap.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. The NMT base model used in CoVe.*

- A sequence of $$n$$ words in source language (English): $$x = [x_1, \dots, x_n]$$.
- A sequence of $$m$$ words in target language (German): $$y = [y_1, \dots, y_m]$$.
- The [GloVe]({{ site.baseurl }}{% post_url 2017-10-15-learning-word-embedding %}#glove-global-vectors) vectors of source words: $$\text{GloVe}(x)$$.
- Randomly initialized embedding vectors of target words: $$z = [z_1, \dots, z_m]$$.
- The biLSTM encoder outputs a sequence of hidden states: $$h = [h_1, \dots, h_n] = \text{biLSTM}(\text{GloVe}(x))$$ and $$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$ where the forward LSTM computes $$\overrightarrow{h}_t = \text{LSTM}(x_t, \overrightarrow{h}_{t-1})$$ and the backward computation gives us $$\overleftarrow{h}_t = \text{LSTM}(x_t, \overleftarrow{h}_{t-1})$$.
- The attentional decoder outputs a distribution over words: $$p(y_t \mid H, y_1, \dots, y_{t-1})$$ where $$H$$ is a stack of hidden states $$\{h\}$$ along the time dimension:


$$
\begin{aligned}
\text{decoder hidden state: } s_t &= \text{LSTM}([z_{t-1}; \tilde{h}_{t-1}], s_{t-1}) \\
\text{attention weights: } \alpha_t &= \text{softmax}(H(W_1 s_t + b_1)) \\
\text{context-adjusted hidden state: } \tilde{h}_t &= \tanh(W_2[H^\top\alpha_t;s_t] + b_2) \\
\text{decoder output: } p(y_t\mid H, y_1, \dots, y_{t-1}) &= \text{softmax}(W_\text{out} \tilde{h}_t + b_\text{out})
\end{aligned}
$$


### Use CoVe in Downstream Tasks

The hidden states of NMT encoder are defined as **context vectors** for other language tasks: 

$$
\text{CoVe}(x) = \text{biLSTM}(\text{GloVe}(x))
$$ 

The paper proposed to use the concatenation of GloVe and CoVe for question-answering and classification tasks. GloVe learns from the ratios of global word co-occurrences, so it has no sentence context, while CoVe is generated by processing text sequences is able to capture the contextual information.

$$
v = [\text{GloVe}(x); \text{CoVe}(x)]
$$

Given a downstream task, we first generate the concatenation of GloVe + CoVe vectors of input words and then feed them into the task-specific models as additional features.

![CoVe model]({{ '/assets/images/CoVe.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. The CoVe embeddings are generated by an encoder trained for machine translation task. The encoder can be plugged into any downstream task-specific model. (Image source: [original paper](https://arxiv.org/abs/1708.00107))*


**Summary**: The limitation of CoVe is obvious: (1) pre-training is bounded by available datasets on the supervised translation task; (2) the contribution of CoVe to the final performance is constrained by the task-specific model architecture.

In the following sections, we will see that ELMo overcomes issue (1) by unsupervised pre-training and OpenAI GPT & BERT further overcome both problems by unsupervised pre-training + using generative model architecture for different downstream tasks.



## ELMo

**ELMo**, short for **Embeddings from Language Model** ([Peters, et al, 2018](https://arxiv.org/abs/1802.05365)) learns contextualized word representation by pre-training a language model in an *unsupervised* way.


### Bidirectional Language Model

The bidirectional Language Model (**biLM**) is the foundation for ELMo. While the input is a sequence of $$n$$ tokens, $$(x_1, \dots, x_n)$$, the language model learns to predict the probability of next token given the history.

In the forward pass, the history contains words before the target token,

$$
p(x_1, \dots, x_n) = \prod_{i=1}^n p(x_i \mid x_1, \dots, x_{i-1})
$$

In the backward pass, the history contains words after the target token,

$$
p(x_1, \dots, x_n) = \prod_{i=1}^n p(x_i \mid x_{i+1}, \dots, x_n)
$$

The predictions in both directions are modeled by multi-layer LSTMs with hidden states $$\overrightarrow{\mathbf{h}}_{i,\ell}$$ and $$\overleftarrow{\mathbf{h}}_{i,\ell}$$ for input token $$x_i$$ at the layer level $$\ell=1,\dots,L$$.
The final layer’s hidden state $$\mathbf{h}_{i,L} = [\overrightarrow{\mathbf{h}}_{i,L}; \overleftarrow{\mathbf{h}}_{i,L}]$$ is used to output the probabilities over tokens after softmax normalization. They share the embedding layer and the softmax layer, parameterized by $$\Theta_e$$ and $$\Theta_s$$ respectively.



![ELMo biLSTM]({{ '/assets/images/ELMo-biLSTM.png' | relative_url }})
{: style="width: 85%;" class="center"}
*Fig. 3. The biLSTM base model of ELMo. (Image source: recreated based on [this](https://towardsdatascience.com/introduction-to-sequence-models-rnn-bidirectional-rnn-lstm-gru-73927ec9df15).)*

The model is trained to minimize the negative log likelihood (= maximize the log likelihood for true words) in both directions:

$$
\begin{aligned}
\mathcal{L} = - \sum_{i=1}^n \Big( 
\log p(x_i \mid x_1, \dots, x_{i-1}; \Theta_e, \overrightarrow{\Theta}_\text{LSTM}, \Theta_s) + \\
\log p(x_i \mid x_{i+1}, \dots, x_n; \Theta_e, \overleftarrow{\Theta}_\text{LSTM}, \Theta_s) \Big)
\end{aligned}
$$


### ELMo Representations

On top of a $$L$$-layer biLM, ELMo stacks all the hidden states across layers together by learning a task-specific linear combination. The hidden state representation for the token $$x_i$$ contains $$2L+1$$ vectors:

$$
R_i = \{ \mathbf{h}_{i,\ell} \mid \ell = 0, \dots, L \}
$$
where $$\mathbf{h}_{0, \ell}$$ is the embedding layer output and $$\mathbf{h}_{i, \ell} = [\overrightarrow{\mathbf{h}}_{i,\ell}; \overleftarrow{\mathbf{h}}_{i,\ell}]$$.

The weights, $$\mathbf{s}^\text{task}$$, in the linear combination are learned for each end task and normalized by softmax. The scaling factor $$\gamma^\text{task}$$ is used to correct the misalignment between the distribution of biLM hidden states and the distribution of task specific representations.

$$
v_i = f(R_i; \Theta^\text{task}) = \gamma^\text{task} \sum_{\ell=0}^L s^\text{task}_i \mathbf{h}_{i,\ell}
$$

To evaluate what kind of information is captured by hidden states across different layers, ELMo is applied on semantic-intensive and syntax-intensive tasks respectively using representations in different layers of biLM:
- **Semantic task**: The *word sense disambiguation (WSD)* task emphasizes the meaning of a word given a context. The biLM top layer is better at this task than the first layer.
- **Syntax task**: The *[part-of-speech](https://en.wikipedia.org/wiki/Part-of-speech_tagging) (POS) tagging* task aims to infer the grammatical role of a word in one sentence. A higher accuracy can be achieved by using the biLM first layer than the top layer.

The comparison study indicates that syntactic information is better represented at lower layers while semantic information is captured by higher layers. Because different layers tend to carry different type of information, *stacking them together helps*.


### Use ELMo in Downstream Tasks

Similar to how [CoVe](#use-cove-in-downstream-tasks) can help different downstream tasks, ELMo embedding vectors are included in the input or lower levels of task-specific models. Moreover, for some tasks (i.e., [SNLI](#nli) and [SQuAD](#qa), but not [SRL](#srl)), adding them into the output level helps too.

The improvements brought up by ELMo are largest for tasks with a small supervised dataset. With ELMo, we can also achieve similar performance with much less labeled data.


**Summary**: The language model pre-training is unsupervised and theoretically the pre-training can be scaled up as much as possible since the unlabeled text corpora are abundant. However, it still has the dependency on task-customized models and thus the improvement is only incremental, while searching for a good model architecture for every task remains non-trivial.



## Cross-View Training

In ELMo the unsupervised pre-training and task-specific learning happen for two independent models in two separate training stages. **Cross-View Training** (abbr. **CVT**; [Clark et al., 2018](https://arxiv.org/abs/1809.08370)) combines them into one unified semi-supervised learning procedure where the representation of a biLSTM encoder is improved by both supervised learning with labeled data and unsupervised learning with unlabeled data on auxiliary tasks.


### Model Architecture

The model consists of a two-layer bidirectional LSTM encoder and a primary prediction module. During training, the model is fed with labeled and unlabeled data batches alternatively.
- On *labeled examples*, all the model parameters are updated by standard supervised learning. The loss is the standard cross entropy.
- On *unlabeled examples*, the primary prediction module still can produce a "soft" target, even though we cannot know exactly how accurate they are. In a couple of auxiliary tasks, the predictor only sees and processes a restricted view of the input, such as only using encoder hidden state representation in one direction. The auxiliary task outputs are expected to match the primary prediction target for a full view of input. <br/>In this way, the encoder is forced to distill the knowledge of the full context into partial representation. At this stage, the biLSTM encoder is backpropagated but the primary prediction module is *fixed*. The loss is to minimize the distance between auxiliary and primary predictions.


![CVT]({{ '/assets/images/CVT.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. The overview of semi-supervised language model cross-view training. (Image source: [original paper](https://arxiv.org/abs/1809.08370))*


### Multi-Task Learning

When training for multiple tasks simultaneously, CVT adds several extra primary prediction models for additional tasks. They all share the same sentence representation encoder.
During supervised training, once one task is randomly selected, parameters in its corresponding predictor and the representation encoder are updated.
With unlabeled data samples, the encoder is optimized jointly across all the tasks by minimizing the differences between auxiliary outputs and primary prediction for every task.

The multi-task learning encourages better generality of representation and in the meantime produces a nice side-product: all-tasks-labeled examples from unlabeled data. They are precious data labels considering that cross-task labels are useful but fairly rare.


### Use CVT in Downstream Tasks

Theoretically the primary prediction module can take any form, generic or task-specific design. The examples presented in the CVT paper include both cases.

In sequential tagging tasks (classification for every token) like [NER](#ner) or [POS](#pos) tagging, the predictor module contains two fully connected layers and a softmax layer on the output to produce a probability distribution over class labels.
For each token $$\mathbf{x}_i$$, we take the corresponding hidden states in two layers, $$\mathbf{h}_1^{(i)}$$ and $$\mathbf{h}_2^{(i)}$$:


$$
\begin{aligned}
p_\theta(y_i \mid \mathbf{x}_i) 
&= \text{NN}(\mathbf{h}^{(i)}) \\
&= \text{NN}([\mathbf{h}_1^{(i)}; \mathbf{h}_2^{(i)}]) \\
&= \text{softmax} \big( \mathbf{W}\cdot\text{ReLU}(\mathbf{W'}\cdot[\mathbf{h}_1^{(i)}; \mathbf{h}_2^{(i)}]) + \mathbf{b} \big)
\end{aligned}
$$

The auxiliary tasks are only fed with forward or backward LSTM state in the first layer. Because they only observe partial context, either on the left or right, they have to learn like a language model, trying to predict the next token given the context. The `fwd` and `bwd` auxiliary tasks only take one direction. The `future` and `past` tasks take one step further in forward and backward direction, respectively.


$$
\begin{aligned}
p_\theta^\text{fwd}(y_i \mid \mathbf{x}_i) &= \text{NN}^\text{fwd}(\overrightarrow{\mathbf{h}}^{(i)}) \\
p_\theta^\text{bwd}(y_i \mid \mathbf{x}_i) &= \text{NN}^\text{bwd}(\overleftarrow{\mathbf{h}}^{(i)}) \\
p_\theta^\text{future}(y_i \mid \mathbf{x}_i) &= \text{NN}^\text{future}(\overrightarrow{\mathbf{h}}^{(i-1)}) \\
p_\theta^\text{past}(y_i \mid \mathbf{x}_i) &= \text{NN}^\text{past}(\overleftarrow{\mathbf{h}}^{(i+1)})
\end{aligned}
$$


![CVT sequential tagging]({{ '/assets/images/CVT-example.png' | relative_url }})
{: style="width: 70%;" class="center"}
*Fig. 5. The sequential tagging task depends on four auxiliary prediction models, their inputs only involving hidden states in one direction: forward, backward, future and past. (Image source: [original paper](https://arxiv.org/abs/1809.08370))*

Note that if the primary prediction module has dropout, the dropout layer works as usual when training with labeled data, but it is not applied when generating "soft" target for auxiliary tasks during training with unlabeled data.

In the machine translation task, the primary prediction module is replaced with a standard unidirectional LSTM decoder with attention. There are two auxiliary tasks: (1) apply dropout on the attention weight vector by randomly zeroing out some values; (2) predict the future word in the target sequence. The primary prediction for auxiliary tasks to match is the best predicted target sequence produced by running the fixed primary decoder on the input sequence with [beam search](https://en.wikipedia.org/wiki/Beam_search).


## OpenAI GPT

Following the similar idea of ELMo, OpenAI **GPT**, short for **Generative Pre-training Transformer** ([Radford et al., 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)), expands the unsupervised language model to a much larger scale by training on a giant collection of free text corpora. Despite of the similarity, GPT has two major differences from ELMo.
1. The model architectures are different: ELMo uses a shallow concatenation of independently trained left-to-right and right-to-left multi-layer LSTMs, while GPT is a multi-layer transformer decoder.
2. The use of contextualized embeddings in downstream tasks are different: ELMo feeds embeddings into models customized for specific tasks as additional features, while GPT fine-tunes the same base model for all end tasks.
    * Generative pre-trained LM + task-specific fine-tuning has been proved to work in [ULMFiT](https://arxiv.org/abs/1801.06146). But in ULMFiT the fine-tuning happens in all layers gradually and ULMFiT focused on training techniques for stabilizing the fine-tuning process.


### Transformer Decoder as Language Model

Compared to the [original transformer](https://arxiv.org/abs/1706.03762) architecture, the [transformer decoder](https://arxiv.org/abs/1801.10198) model discards the encoder part, so there is only one single input sentence rather than two separate source and target sequences.
            
This model applies multiple transformer blocks over the embeddings of input sequences. Each block contains a masked *multi-headed self-attention* layer and a *pointwise feed-forward* layer. The final output produces a distribution over target tokens after softmax normalization.


![OpenAI GPT transformer decoder]({{ '/assets/images/OpenAI-GPT-transformer-decoder.png' | relative_url }})
{: style="width: 85%;" class="center"}
*Fig. 6. The transformer decoder model architecture in OpenAI GPT.*

The loss is the negative log-likelihood, same as [ELMo](#elmo), but without backward computation. Let’s say, the context window of the size $$k$$ is located before the target word and the loss would look like:


$$
\mathcal{L}_\text{LM} = -\sum_{i} \log p(x_i\mid x_{i-k}, \dots, x_{i-1})
$$


### Supervised Fine-Tuning

The most substantial upgrade that OpenAI GPT proposed is to get rid of the task-specific model and use the pre-trained language model directly!

Let’s take classification as an example. Say, in the labeled dataset, each input has $$n$$ tokens, $$\mathbf{x} = (x_1, \dots, x_n)$$, and one label $$y$$. GPT first processes the input sequence $$\mathbf{x}$$ through the pre-trained transformer decoder and the last layer output for the last token $$x_n$$ is $$\mathbf{h}_L^{(n)}$$. Then with only one new trainable weight matrix $$\mathbf{W}_y$$, it can predict a distribution over class labels.


![GPT classification]({{ '/assets/images/GPT-classification.png' | relative_url }})
{: style="width: 90%;" class="center"}



$$
P(y\mid x_1, \dots, x_n) = \text{softmax}(\mathbf{h}_L^{(n)}\mathbf{W}_y)
$$

The loss is to minimize the negative log-likelihood for true labels. In addition, adding the LM loss as an auxiliary loss is found to be beneficial, because:
- (1) it helps accelerate convergence during training and 
- (2) it is expected to improve the generalization of the supervised model.

$$
\begin{aligned}
\mathcal{L}_\text{cls} &= \sum_{(\mathbf{x}, y) \in \mathcal{D}} \log P(y\mid x_1, \dots, x_n) = \sum_{(\mathbf{x}, y) \in \mathcal{D}} \log \text{softmax}(\mathbf{h}_L^{(n)}(\mathbf{x})\mathbf{W}_y) \\
\mathcal{L}_\text{LM} &= -\sum_{i} \log p(x_i\mid x_{i-k}, \dots, x_{i-1}) \\
\mathcal{L} &= \mathcal{L}_\text{cls} + \lambda \mathcal{L}_\text{LM}
\end{aligned}
$$

With similar designs, no customized model structure is needed for other end tasks (see Fig. 7). If the task input contains multiple sentences, a special delimiter token (`$`) is added between each pair of sentences. The embedding for this delimiter token is a new parameter we need to learn, but it should be pretty minimal. 

For the sentence similarity task, because the ordering does not matter, both orderings are included. For the multiple choice task, the context is paired with every answer candidate.


![GPT downstream tasks]({{ '/assets/images/GPT-downstream-tasks.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Training objects in slightly modified GPT transformer models for downstream tasks. (Image source: [original paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf))*


**Summary**: It is super neat and encouraging to see that such a general framework is capable to beat SOTA on most language tasks at that time (June 2018). At the first stage, generative pre-training of a language model can absorb as much free text as possible. Then at the second stage, the model is fine-tuned on specific tasks with a small labeled dataset and a minimal set of new parameters to learn. 

One limitation of GPT is its uni-directional nature --- the model is only trained to predict the future left-to-right context.


## BERT

**BERT**, short for **Bidirectional Encoder Representations from Transformers** ([Devlin, et al., 2019](https://arxiv.org/abs/1810.04805)) is a direct descendant to [GPT](#gpt): train a large language model on free text and then fine-tune on specific tasks without customized network architectures.

Compared to GPT, the largest difference and improvement of BERT is to make training **bi-directional**. The model learns to predict both context on the left and right. The paper according to the ablation study claimed that:

> "bidirectional nature of our model is the single most important new contribution"


### Auxiliary Tasks

The model architecture of BERT is a multi-layer bidirectional Transformer encoder.

![transformer encoder]({{ '/assets/images/transformer-encoder-2.png' | relative_url }})
{: style="width: 25%;" class="center"}
*Fig. 8. Recap of Transformer Encoder model architecture. (Image source: [Transformer paper](https://arxiv.org/abs/1706.03762))*

To encourage the bi-directional prediction and sentence-level understanding, BERT applied two auxiliary tasks in addition to the basic language task (that is, to predict the next token given context).

**Task 1: Mask language model (MLM)**

> From [Wikipedia](https://en.wikipedia.org/wiki/Cloze_test): "A cloze test (also cloze deletion test) is an exercise, test, or assessment consisting of a portion of language with certain items, words, or signs removed (cloze text), where the participant is asked to replace the missing language item. … The exercise was first described by W.L. Taylor in 1953."

It is unsurprising to believe that a representation that learns the context around a word rather than just after the word is able to better capture its meaning, both syntactically and semantically. BERT encourages the model to do so by training on the *"mask language model" task*:
1. Randomly mask 15% of tokens in each sequence. Because if we only replace masked tokens with a special placeholder `[MASK]`, the special token would never be encountered during fine-tuning. Hence, BERT employed several heuristic tricks:
    - (a) with 80% probability, replace the chosen words with `[MASK]`;
    - (b) with 10% probability, replace with a random word;
    - (c) with 10% probability, keep it the same.
2. The model only predicts the missing words, but it has no information on which words have been replaced or which words should be predicted. The output size is only 15% of the input size. 

**Task 2: Next sentence prediction**

Motivated by the fact that many downstream tasks involve the understanding of relationships between sentences (i.e., [QA](#qa), [NLI](#nli)), BERT added another auxiliary task on training a *binary classifier* for telling whether one sentence is the next sentence of the other:
1. Sample sentence pairs (A, B) so that:
    - (a) 50% of the time, B follows A;
    - (b) 50% of the time, B does not follow A.
2. The model processes both sentences and output a binary label indicating whether B is the next sentence of A.

The training data for both auxiliary tasks above can be trivially generated from any monolingual corpus. Hence the scale of training is unbounded. The training loss is the sum of the mean masked LM likelihood and mean next sentence prediction likelihood.


![Language model comparison]({{ '/assets/images/language-model-comparison.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 9. Comparison of BERT, OpenAI GPT and ELMo model architectures. (Image source: [original paper](https://arxiv.org/abs/1810.04805))*


### Input Embedding

The input embedding is the sum of three parts:
1. WordPiece tokenization embeddings: The WordPiece [model](https://arxiv.org/pdf/1609.08144.pdf) was originally proposed for Japanese or Korean segmentation problem. Instead of using naturally split English word, they can be further divided into smaller sub-word units so that it is more effective to handle rare or unknown words. Please read [linked](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) papers for the optimal way to split words if interested.
2. Segment embeddings: If the input contains two sentences, they have sentence A embeddings and sentence B embeddings respectively and they are separated by a special character `[SEP]`; Only sentence A embeddings are used if the input contains one sentence.
3. Position embeddings: Positional embeddings are learned.


![BERT input embedding]({{ '/assets/images/BERT-input-embedding.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 10. BERT input representation. (Image source: [original paper](https://arxiv.org/abs/1810.04805))*


Note that the first token is always forced to be `[CLS]`, a placeholder that will be used later for prediction in downstream tasks.



### Comparison between OpenAI GPT and BERT

A summary table compares differences between the training of OpenAI GPT and BERT.

{: class="info"}
|              | **OpenAI GPT** | **BERT** |
| Special char | `[SEP]` and `[CLS]` are only introduced at fine-tuning stage. | `[SEP]` and `[CLS]` and sentence A/B embeddings are learned at the pre-training stage. |
| Training process | 1M steps, batch size 32k words. | 1M steps, batch size 128k words. |
| Fine-tuning  | lr = 5e-5 for all fine-tuning tasks. | Use task-specific lr for fine-tuning. |



### Use BERT in Downstream Tasks

BERT fine-tuning requires only a few new parameters added, just like OpenAI GPT.

For classification tasks, we get the prediction by taking the final hidden state of the special first token `[CLS]`, $$\mathbf{h}^\text{[CLS]}_L$$, and multiplying it with a small weight matrix, $$\text{softmax}(\mathbf{h}^\text{[CLS]}_L \mathbf{W}_\text{cls})$$. 


For QA tasks like SQuAD, we need to predict the text span in the given paragraph for an given question. BERT predicts two probability distributions of every token, being the start and the end of the text span. Only two new small matrices, $$\mathbf{W}_\text{s}$$ and $$\mathbf{W}_\text{e}$$, are newly learned during fine-tuning and $$\text{softmax}(\mathbf{h}^\text{(i)}_L \mathbf{W}_\text{s})$$ and $$\text{softmax}(\mathbf{h}^\text{(i)}_L \mathbf{W}_\text{s})$$ define the probability distributions.

Overall the add-on parts for end task fine-tuning are very minimal - one or two weight matrices to convert the Transform hidden states to an interpretable format. Check the paper for implementation details for other cases.


![BERT downstream tasks]({{ '/assets/images/BERT-downstream-tasks.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 11. Training objects in slightly modified BERT models for downstream tasks.  (Image source: [original paper](https://arxiv.org/abs/1810.04805))*



## Summary

{: class="info"}
|     | Base model | pre-training | Downstream tasks | Downstream model | Fine-tuning |
| --- | --- | --- | --- | --- | --- |
| CoVe | seq2seq NMT model | supervised | feature-based | task-specific | / |
| ELMo | two-layer biLSTM | unsupervised | feature-based | task-specific | / |
| CVT | two-layer biLSTM | semi-supervised | model-based | task-specific / task-agnostic | / |
| ULMFiT | AWD-LSTM | unsupervised | model-based | task-agnostic | all layers; with various training tricks
| OpenAI GPT | Transformer decoder | unsupervised | model-based | task-agnostic | only the top task layer(s) |
| BERT | Transformer encoder | unsupervised | model-based | task-agnostic | only the top task layer(s) |


## Common Tasks and Datasets

<a name='qa' />
**Question-Answering**
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Stanford Question Answering Dataset): 
- [RACE](http://www.qizhexie.com/data/RACE_leaderboard) (ReAding Comprehension from Examinations): A large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students.


**Commonsense Reasoning**
- [Story Cloze Test](http://cs.rochester.edu/nlp/rocstories/): A commonsense reasoning framework for evaluating story understanding and generation. The test requires a system to choose the correct ending to multi-sentence stories from two options.
- [SWAG](https://rowanzellers.com/swag/) (Situations With Adversarial Generations): multiple choices; contains 113k sentence-pair completion examples that evaluate grounded common-sense inference

<a name='nli' />
**Natural Language Inference (NLI)**: also known as **Text Entailment**, an exercise to discern in logic whether one sentence can be inferred from another. 
- [RTE](https://aclweb.org/aclwiki/Textual_Entailment_Resource_Pool) (Recognizing Textual Entailment): A set of datasets initiated by text entailment challenges.
- [SNLI](https://nlp.stanford.edu/projects/snli/) (Stanford Natural Language Inference): A collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels `entailment`, `contradiction`, and `neutral`.
- [MNLI](https://www.nyu.edu/projects/bowman/multinli/) (Multi-Genre NLI): Similar to SNLI, but with a more diverse variety of text styles and topics, collected from transcribed speech, popular fiction, and government reports.
- [QNLI](https://gluebenchmark.com/tasks) (Question NLI): Converted from SQuAD dataset to be a binary classification task over pairs of (question, sentence).
- [SciTail](http://data.allenai.org/scitail/): An entailment dataset created from multiple-choice science exams and web sentences.


<a name='ner' />
**Named Entity Recognition (NER)**: labels sequences of words in a text which are the names of things, such as person and company names, or gene and protein names
- [CoNLL 2003 NER task](https://www.clips.uantwerpen.be/conll2003/): consists of newswire from the Reuters, concentrating on four types of named entities: persons, locations, organizations and names of miscellaneous entities.
- [OntoNotes 0.5](https://catalog.ldc.upenn.edu/LDC2013T19): This corpus contains text in English, Arabic and Chinese, tagged with four different entity types (PER, LOC, ORG, MISC).
- [Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html): A large collection of Reuters News stories.
- Fine-Grained NER (FGN)


**Sentiment Analysis**
- [SST](https://nlp.stanford.edu/sentiment/index.html) (Stanford Sentiment Treebank)
- [IMDb](http://ai.stanford.edu/~amaas/data/sentiment/): A large dataset of movie reviews with binary sentiment classification labels.


<a name='srl' />
**Semantic Role Labeling (SRL)**: models the predicate-argument structure of a sentence, and is often described as answering "Who did what to whom".
- [CoNLL-2004 & CoNLL-2005](http://www.lsi.upc.edu/~srlconll/)


**Sentence similarity**: also known as *paraphrase detection*
- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) (MicRosoft Paraphrase Corpus): It contains pairs of sentences extracted from news sources on the web, with annotations indicating whether each pair is semantically equivalent.
- [QQP](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) (Quora Question Pairs)
STS Benchmark: Semantic Textual Similarity


**Sentence Acceptability**: a task to annotate sentences for grammatical acceptability.
- [CoLA](https://nyu-mll.github.io/CoLA/) (Corpus of Linguistic Acceptability): a binary single-sentence classification task.


**Text Chunking**: To divide a text in syntactically correlated parts of words.
- [CoNLL-2000](https://www.clips.uantwerpen.be/conll2000/chunking/): 


<a name='pos' />
**Part-of-Speech (POS) Tagging**: tag parts of speech to each token, such as noun, verb, adjective, etc.
the Wall Street Journal portion of the Penn Treebank (Marcus et al., 1993).


**Machine Translation**:  See [Standard NLP](https://nlp.stanford.edu/projects/nmt/) page.
- WMT 2015 English-Czech data [Large]
- WMT 2014 English-German data [Medium]
- IWSLT 2015 English-Vietnamese data [Small]


**Coreference Resolution**: cluster mentions in text that refer to the same underlying real world entities.
- [CoNLL-2012](http://conll.cemantix.org/2012/data.html)


**GLUE** multi-task benchmark: [https://gluebenchmark.com](https://gluebenchmark.com/) 


**Unsupervised pre-training dataset**
- [Books corpus](https://googlebooks.byu.edu/): The corpus contains "over 7,000 unique unpublished books from a variety of genres including Adventure, Fantasy, and Romance."
- [1B Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/)
- [English Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Database_download#English-language_Wikipedia): ~2500M words


## Reference

[1] Bryan McCann, et al. ["Learned in translation: Contextualized word vectors."](https://arxiv.org/abs/1708.00107) NIPS. 2017.

[2] Kevin Clark et al. ["Semi-Supervised Sequence Modeling with Cross-View Training."](https://arxiv.org/abs/1809.08370) EMNLP 2018.

[3] Matthew E. Peters, et al. ["Deep contextualized word representations."](https://arxiv.org/abs/1802.05365) NAACL-HLT 2017.

[4] OpenAI Blog ["Improving Language Understanding with Unsupervised Learning"](https://blog.openai.com/language-unsupervised/), June 11, 2018.

[5] Jeremy Howard and Sebastian Ruder. ["Universal language model fine-tuning for text classification."](https://arxiv.org/abs/1801.06146) ACL 2018.

[6] Alec Radford et al. ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI Blog, June 11, 2018.

[7] Jacob Devlin, et al. ["BERT: Pre-training of deep bidirectional transformers for language understanding."](https://arxiv.org/abs/1810.04805) arXiv:1810.04805 (2018).

[8] Mike Schuster, and Kaisuke Nakajima. ["Japanese and Korean voice search."](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) ICASSP. 2012.

[9] Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation

[10] Ashish Vaswani, et al. ["Attention is all you need."](https://arxiv.org/abs/1706.03762) NIPS 2017.

[11] Peter J. Liu, et al. ["Generating wikipedia by summarizing long sequences."](https://arxiv.org/abs/1801.10198) ICLR 2018.

[12] Sebastian Ruder. ["10 Exciting Ideas of 2018 in NLP"](http://ruder.io/10-exciting-ideas-of-2018-in-nlp/) Dec 2018.
