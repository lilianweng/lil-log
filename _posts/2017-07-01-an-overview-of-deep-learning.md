---
layout: post
title: "An Overview of Deep Learning for Curious People"
date: 2017-07-06 01:09:00
---

> Starting earlier this year, I grew a strong curiosity of deep learning and spent some time reading about the field. To document what I’ve learned and to provide some interesting pointers to people with similar interests, I wrote this overview of deep learning models and their applications.


<!--more-->

(The post was originated from my talk for [WiMLDS x Fintech meetup](http://wimlds.org/chapters/about-bay-area/) hosted by [Affirm](www.affirm.com).)

**Table of Content**
* TOC
{:toc}

---

I believe many of you have watched or heard of the [games](https://youtu.be/vFr3K2DORc8) between AlphaGo and professional Go player Lee Sedol in 2006. Lee has the highest rank of nine dan and many world championships. No doubt, he is one of the best professional Go players, but he [lost by 1-4](https://www.scientificamerican.com/article/how-the-computer-beat-the-go-master/) in this series versus AlphaGo. Before this, Go was considered to be an intractable game for computers to master, as its simple rules lay out an exponential number of variation in the board positions, many more than what in Chess. This event definitely highlighted 2016 as a big year for AI. Because of AlphaGo, much attention was attracted to the progress of AI.


Meanwhile, many companies are spending resources on pushing the edges of various applications, that indeed have the potential to completely change or even revolutionize how we are gonna live. Familiar examples include self-driving cars, chatbots, home assistant devices and many others. One of the secret receipts behind the progress we have had in recent years is deep learning.


## Why Does Deep Learning Works Now?

Deep learning models, in simple words, are large and deep artificial neural nets. A neural network (“NN”) can be well presented in a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph): the input layer takes in signal vectors; one or multiple hidden layers process the outputs of the previous layer. The initial concept of neural network can be traced back to more than [half a century ago](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/History/history1.html). But why does it work now? Why do people start talking about them all of a sudden? 

![Artificial neural network](/assets/images/ANN.png)
{: style="width: 400px"}
*Fig 1. A three-layer artificial neural network. (Image source: <http://cs231n.github.io/convolutional-networks/#conv>)*


The reason is surprisingly simple:
- We have a lot **more data**.
- We have much **powerful computers**.

A large and deep neural network has many more layers + many more nodes in each layer, which results in exponentially many more parameters to tune. Without enough data, we cannot learn parameters efficiently. Without powerful computers, learning would be too slow and insufficient.


Here is an interesting plot presenting the relationship between the data scale and the model performance, proposed by Andrew Ng in his "[Nuts and Bolts of Applying Deep Learning](https://youtu.be/F1ka6a13S9I)" talk. On a small dataset, traditional algorithms (Regression, Random Forests, SVM, GBM, etc.) or statistical learning can a great job, but once the data scale goes up to the sky, the large NN outperforms others. Partially because NN has many more parameters and has the capability to learn complicated nonlinear patterns. Thus we expect the model is to pick the most helpful features by itself without too much expert-involved manual feature engineering.


![Data size versus model performance](/assets/images/data_size_vs_model_performance.png)
{: style="width: 400px"}
*Fig 2: Data size versus the model performance. (Recreated based on: <https://youtu.be/F1ka6a13S9I>)*


## Deep Learning Models

Next let's go through a few classical deep learning models. 


### Convolutional Neural Network

Convolutional neural networks, short for "CNN", is a type of feed-forward artificial neural network, in which the connectivity pattern between its neurons is inspired by the organization of the visual cortex system. The primary visual cortex (V1) does edge detection, out of the raw input from retina. The secondary visual cortex (V2), also called prestriate cortex, takes in the edge features from V1 and extract simple visual properties like orientation, spatial frequency, color, etc. The visual area V4 can handle more complicated object features. All the processed visual features go into the final logic unit, inferior temporal gyrus (IT) for object recognition. The shortcut between V1 and V4 inspires a special type of CNN with connections between non-adjacent layers: Residual Net ([He, et al. 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)) with "Residual Block" which supports some input of one layer can be passed to the component two layers later. 


![Human visual cortex system](/assets/images/visual_cortex_system.png)
{: style="width: 680px"}
*Fig 3: Illustration of human visual cortex system.(The source of the left image: Source: Wang, Haohan, Bhiksha Raj, and Eric P. Xing. ["On the Origin of Deep Learning."](https://arxiv.org/pdf/1702.07800.pdf) arXiv preprint arXiv:1702.07800, 2017.)*


Convolution is a mathematical term, here referring to an operation between two matrices. The convolutional layer has a fixed small matrix defined, also called kernel or filter. As the kernel is sliding, or convolving, across the matrix representation of the input image, it is computing the element-wise multiplication of the values in the kernel matrix with the original image values. [Specially designed kernels](http://setosa.io/ev/image-kernels/) can process images for common purposes like blur, sharpen, edge detection and many more, fast and efficiently.


![Architecture of LeNet](/assets/images/lenet.png)
*Fig 4: The LeNet architecture consists of two sets of convolutional, activation, and pooling layers, followed by a fully-connected layer, activation, another fully-connected, and finally a softmax classifier (Image source: <http://deeplearning.net/tutorial/lenet.html>)*


[Convolutional](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/) and [pooling](http://ufldl.stanford.edu/tutorial/supervised/Pooling/) layers act like the primary / secondary visual cortices, working on feature extraction. The objection recognition reasoning happens in later fully connected neuron layers which consumes the extracted features.


### Recurrent Neural Network

An sequence model is usually designed to turn an input sequence into an output sequence that lives in a different domain. Recurrent Neural Network, short for "RNN", fits in this purpose naturally and has shown tremendous improvement in handwriting recognition, speech recognition, and machine translation ([Sutskever et al. 2011](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Sutskever_524.pdf), [Liwicki et al. 2007](http://www6.in.tum.de/Main/Publications/Liwicki2007a.pdf)).


A recurrent neural network model is designed to handle long sequential data and tasks with context spreading in time. The model processes one element in the sequence at one time step and then passes down the current unit state to the next time step to facilitate the computation of the next element. Imagine the case when a RNN model reads all the Wikipedia articles and then it can predict the following characters.


![Recurrent neural network](/assets/images/RNN.png)
{: style="width: 500px"}
*Fig 5. A recurrent neural network with one hidden unit (left) and its unrolling version in time (right). The unrolling version illustrates what happened in time: s_{t-1}, s_{t}, and s_{t+1} are the same unit with different states at time step $$ t-1 $$, $$ t $$, and $$ t+1 $$. (Image source: <https://www.nature.com/nature/journal/v521/n7553/fig_tab/nature14539_F5.html>)*


However, simple perceptron neurons that combine the current input element and the last unit state can lose the long-term dependencies easily. For example, we start a sentence with “Alice is working at…”, and after a whole paragraph, we want to start the next sentence with “She” or “He” correctly. If the model forgets the character’s name “Alice”, we can never know. To resolve the issue, researchers created a special neuron with much more complicated internal structure for memorizing long-term context, named ["Long-short term memory (LSTM)"](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf) cell. It is smart enough to learn for how long it should memorize the old information, when to forget, when to make use of the new input, and how to combine the old memory with new input. This [introduction](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is so well written that I recommend everyone with interest in LSTM to read it. It is officially promoted in the [Tensorflow documentation](https://www.tensorflow.org/tutorials/recurrent) ;)


![LSTM](/assets/images/LSTM.png)
{: style="width: 320px"}
*Fig 6. The structure of a LSTM cell. (Image source: <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>)*


To demonstrate the power of RNNs, [Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) built a character-based language model using RNN with LSTM cells.  Without knowing any English vocabulary beforehand, the model could learn the relationship between characters to form words and then the relationship between words to form sentences. It could achieve a decent performance even without a huge set of training data.


![Shakespeare by RNN](/assets/images/rnn_shakespeare.png)
{: style="width: 500px"}
*Fig 7. A character-based RNN model can write like a Shakespeare. (Image source: <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>)*


### RNN: Sequence-to-Sequence Model

The [sequence-to-sequence model](https://arxiv.org/pdf/1406.1078.pdf) is a type of RNN models, but its application field is distinguishable enough that I would like to list it in a separated section. It is commonly used to develop chatbots or personal assistants, both generating meaningful response for input questions. A sequence-to-sequence model consists of two RNNs, encoder and decoder. The encoder consumes the input words and passes the knowledge and the contextual information it has learnt to the decoder through a context vector (the "thought vector" in Fig 8.). The decoder takes in the context vector and generates proper responses.
 

![Sequence-to-sequence model](/assets/images/seq2seq_gmail.png)
*Fig 8. A sequence-to-sequence model for generating Gmail auto replies. (Image source: <https://research.googleblog.com/2015/11/computer-respond-to-this-email.html>)*


### Autoencoders

Different from the previous models, autoencoders are unsupervised learning. It is designed to learn a lower-dimensional representation of high-dimensional data, similar to what Principal Components Analysis (PCA) does. The autoencoder model tries to learn an approximation function $$ f(x) \approx x $$ to reproduce the data itself. However, it is restricted by a bottleneck layer in the middle with much fewer nodes. Because there are so few units there, the model is forced to learn a very efficient representation of the data, that is essentially the low-dimensional code we learnt.


![Autoencoder](/assets/images/autoencoder.png)
{: style="width: 300px"}
*Fig 9. An autoencoder model has a bottleneck layer with a very few number of nodes. (Image source: Geoffrey Hinton’s Coursera class "Neural Networks for Machine Learning" - [Week 15](https://www.coursera.org/learn/neural-networks/home/week/15))*


[Hinton and Salakhutdinov](https://pdfs.semanticscholar.org/7d76/b71b700846901ac4ac119403aa737a285e36.pdf) used autocoders to compress documents that cover a variety of topics. As shown in Fig 10, when both PCA and autocoders were asked to compress the documents into two dimensions, autocoders demonstrated a much better outcome. With the help of autoencoders, we can do efficient data compression to speed up the information retrieval including both documents and images.


![Autoencoder experiment](/assets/images/autoencoder_experiment.png)
*Fig 10. The outputs of PCA (left) and autoencoder (right) when both try to compress documents into two numbers. (Image source: Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. ["Reducing the dimensionality of data with neural networks."](https://pdfs.semanticscholar.org/7d76/b71b700846901ac4ac119403aa737a285e36.pdf) science 313.5786 (2006): 504-507.)*


## Reinforcement (Deep) Learning

Since I started my post with AlphaGo, let us dig a bit more here on why AlphaGo worked out. Reinforcement learning (“RL”) is one of the secrets behind its success. RL is a subfield of machine learning, which allows machines and software agents to automatically determine the optimal behaviour within a given context, with a goal to maximize the long-term performance measured by a given metric.


![AlphaGo paper](/assets/images/alphago_paper.png)

![AlphaGo model](/assets/images/alphago_model.png)
{: style="width: 600px"}
*Fig 11. AlphaGo neural network training pipeline and architecture. (Image source: [Silver, David, et al.](http://web.iitd.ac.in/~sumeet/Silver16.pdf))*


The AlphaGo system starts with supervised learning to train a fast rollout policy and a policy network: relying on the manually curated training dataset collected from professional players’ games, it learns what is the best strategy given the current position on the game board. Then it applies reinforcement learning by setting up self-play games: the reinforcement learning policy network gets improved when it wins more and more games against previous versions of the policy network. In the self-play stage, AlphaGo becomes stronger and stronger by playing against itself without requiring additional training data.


### Generative Adversarial Network

[Generative adversarial network](https://arxiv.org/pdf/1406.2661.pdf), short for "GAN", is a type of deep generative models. GAN can generate new examples after learning through the real data.  It is consist of two models competing against each other in a zero-sum game framework. The famous deep learning researcher Yann LeCun gave it a super high praise: Generative Adversarial Network is the most interesting idea in the last ten years in machine learning. (https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning)
 

![Generative adversarial network](/assets/images/GAN.png)
{: style="width: 600px"}
*Fig. 12. Architecture of a generative adversarial network. (Image source: <http://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html>)*


In its birth paper, GAN was proposed to generate meaningful images after learning from real photos. The generator model produces fake images and sends the outputs the discriminator model. The discriminator works like a judge, as it is optimized for identifying the real photos from the fake ones. The generator model is trying hard to cheat the discriminator while the judge model is trying hard not to be cheated. This interesting zero-sum game between these two models motivates both to improve their functionalities. Eventually we take the generator model for producing new images.


## Toolkits and Libraries

After learning all those models, you may start wondering how you can implement them and use them. Fortunately we have many open source tools and libraries for building deep learning model. Tensorflow is fairly new but attracts a lot of popularity. It turns out, TensorFlow was the most forked Github project of 2015. All that in a period of 2 months of its release in Nov 2015


![Deep learning toolkit](/assets/images/deep_learning_toolkits.png)
{: style="padding-bottom: 10px"}


## How to Learn

If you are very new to the field and willing to devote some time to studying deep learning in a more systematic way, I would recommend you to start with the book [Deep Learning](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?s=books&ie=UTF8&qid=1499413305&sr=1-1&keywords=deep+learning) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. The Coursera course "[Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks)" by Geoffrey Hinton ([Godfather of deep learning!](https://youtu.be/uAu3jQWaN6E)). The content for the course was prepared around 2006, pretty old, but it help you build up a very solid foundation for understanding deep learning models and further exploration.


Meanwhile, maintain your curiosity and passion. The field is making progress everyday, even classical models may have just been proposed 1-2 years ago. Reading academic papers can help you learn stuffs in depth and keep up with the cutting edge findings.


#### Useful resources:
- Google Scholar: <http://scholar.google.com>
- arXiv cs section: <https://arxiv.org/list/cs/recent>
- Data Science Weekly
- KDnuggets
- Tons of blog posts and online tutorials
- Cousera courses


#### Blog posts mentioned / interesting blogs worthy of checking
- [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers)
- [www.wildml.com](http://www.wildml.com)
- [colah.github.io](http://colah.github.io/)
- [karpathy.github.io](http://karpathy.github.io/)
- [blog.openai.com](https://blog.openai.com)

#### Papers mentioned

[1] He, Kaiming, et al. ["Deep residual learning for image recognition."](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) Proc. IEEE Conf. on computer vision and pattern recognition. 2016.
 
[2] Wang, Haohan, Bhiksha Raj, and Eric P. Xing. ["On the Origin of Deep Learning."](https://arxiv.org/pdf/1702.07800.pdf) arXiv preprint arXiv:1702.07800, 2017.

[3] Sutskever, Ilya, James Martens, and Geoffrey E. Hinton. ["Generating text with recurrent neural networks."](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Sutskever_524.pdf) Proc. of the 28th Intl. Conf. on Machine Learning (ICML). 2011. 
 
[4] Liwicki, Marcus, et al. ["A novel approach to on-line handwriting recognition based on bidirectional long short-term memory networks."](http://www6.in.tum.de/Main/Publications/Liwicki2007a.pdf) Proc. of 9th Intl. Conf. on Document Analysis and Recognition. 2007.

[5] LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. ["Deep learning."](http://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf) Nature 521.7553 (2015): 436-444.

[6] Hochreiter, Sepp, and Jurgen Schmidhuber. ["Long short-term memory."](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf) Neural computation 9.8 (1997): 1735-1780.

[7] Cho, Kyunghyun. et al. ["Learning phrase representations using RNN encoder-decoder for statistical machine translation."](https://arxiv.org/pdf/1406.1078.pdf) Proc. Conference on Empirical Methods in Natural Language Processing 1724–1734 (2014).

[8] Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. ["Reducing the dimensionality of data with neural networks."](https://pdfs.semanticscholar.org/7d76/b71b700846901ac4ac119403aa737a285e36.pdf) science 313.5786 (2006): 504-507.
 
[9] Silver, David, et al. ["Mastering the game of Go with deep neural networks and tree search."](http://web.iitd.ac.in/~sumeet/Silver16.pdf) Nature 529.7587 (2016): 484-489.

[10] Goodfellow, Ian, et al. ["Generative adversarial nets."](https://arxiv.org/pdf/1406.2661.pdf) NIPS, 2014.


