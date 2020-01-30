---
layout: post
comments: true
title: "Curriculum for Reinforcement Learning"
date: 2020-01-29 18:00:00
tags: reinforcement-learning generative-model meta-learning
---


 
> Curriculum is an efficient tool for humans to learn from simple concepts to hard problems. It breaks down complex knowledge by providing a path of learning steps of increasing difficulty. In this post we will examine how the idea of curriculum can help reinforcement learning models learn complicated tasks.
 

<!--more-->


We probably will never succeed to teach integral or derivative to a 3-year-old who does not even know basic arithmetic. That's why education is important, as it provides a systematic way to break down complex knowledge and a nice curriculum for teaching concepts from simple to hard. Curriculum makes learning difficult things easier and approachable for humans. How about teaching our machine learning models with curriculum? Can machine learning algorithms benefit from a smartly designed curriculum?
 
Back in 1993, Jeffrey Elman has proposed the idea of training neural networks with a curriculum. His early work on learning simple language grammar demonstrated the importance of such a strategy: starting with a restricted set of simple data and gradually increasing the complexity of training samples; otherwise the model was not able to learn at all.
 
Compared to training without curriculum, we would expect the adoption of curriculum expedite convergence speed and may or may not improve the final model performance on plateau. To design a good and effective curriculum is not easy and bad curriculum may even hamper learning. Next we will see curriculum learning to be applied on both supervised learning and reinforcement learning; especially in RL, quite a natural fit.
 
In the "The importance of starting small" paper (Elman 1993), I especially like the starting sentences and find them both inspiring and affecting:
 
> "Humans differ from other species along many dimensions, but two are particularly noteworthy. Humans display an exceptional capacity to learn; and humans are remarkable for the unusually long time it takes to reach maturity. The adaptive advantage of learning is clear, and it may be argued that, through culture, learning has created the basis for a non-genetically based transmission of behaviors which may accelerate the evolution of our species."
 
Indeed, learning is probably the best super power we humans have.
 
Next we will look into several categories of curriculum learning for reinforcement learning, as shown in Fig. 1.
 

![Types of curriculum]({{ '/assets/images/types-of-curriculum.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. Four types of curriculum for reinforcement learning.*


{: class="table-of-content"}
* TOC
{:toc}


## Task-Specific Curriculum

[Bengio, et al. (2009)](https://www.researchgate.net/profile/Y_Bengio/publication/221344862_Curriculum_learning/links/546cd2570cf2193b94c577ac/Curriculum-learning.pdf) provided a good overview of curriculum learning in the old days. The paper presented two ideas with toy experiments using manually designed task-specific curriculum:
1. Cleaner Examples may yield better generalization faster.
2. Introducing gradually more difficult examples speeds up online training.

It is plausible that some curriculum strategies could be useless or even harmful. A good question to answer in the field is: what are the general principles that make some curriculum strategies work better than others? The Bengio 2009 paper hypothesized it would be beneficial to make learning focus on "interesting" examples that are neither too hard or too easy.

If our naive curriculum is to train the model on samples with a gradually increasing level of complexity, we need a way to quantify the difficulty of a task first. One idea is to use its minimal loss with respect to another model while this model is pretrained on different tasks ([Weinshall, et al. 2018](https://arxiv.org/abs/1802.03796)). In this way the knowledge of another model is transferred to the new model by suggesting a rank of training samples. Fig. X shows the effectiveness of `curriculum`, compared to `control` (random order) and `anti` (reverse the order) groups.


![Curriculum by transfer learning]({{ '/assets/images/curriculum-by-transfer-learning.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. Image classification accuracy on test image set (5 member classes of "small mammals" in CIFAR100). There are 4 experimental groups, (a) `curriculum`: sort the labels by the confidence of another trained classifier (e.g. the margin of a SVM); (b) `control-curriculum`: sort the labels randomly; (c) `anti-curriculum`: sort the labels reversely; (d) `None`: xxx. (Image source: [Weinshall, et al. 2018](https://arxiv.org/abs/1802.03796))*

[Zaremba & Sutskever (2014)](https://arxiv.org/abs/1410.4615) did an interesting experiment on training LSTM to predict the output of a short Python program for mathematical ops without actually executing the code. They found curriculum is necessary for learning. The program complexity is controlled by two parameters, `length` ∈ [1, a] and `nesting`∈ [1, b]. Three strategies are considered:
1. Naive curriculum: increase `length` first until reaching `a`; then increase `nesting` and reset `length` to 1; repeat this process until both reach maximum.
2. Mix curriculum: sample `length` ~ [1, a] and `nesting` ~ [1, b]
3. Combined: naive + mix.

They noticed that combined strategy always outperformed naive and would generally (but not always) outperform the mix strategy --- indicating that it is quite important to mix in easy tasks during training to avoid forgetting.
 
To follow the curriculum learning approach in the early days, as described above, generally we need to figure out two problems in the training procedure:
1. Design a metric for measuring how hard a task is so that we can sort tasks accordingly.
2. Provide a sequence of tasks with an increasing level of difficulty to the model during training.

However, the order of tasks does not have to be sequential. In our Rubik's cube paper ([OpenAI et al, 2019](https://arxiv.org/abs/1910.07113.)), we depended on *Automatic domain randomization* (**ADR**) to generate a curriculum by growing a distribution of environments with increasing complexity. The difficulty of each task (i.e. solving a Rubik's cube in a set of environment) depends on the randomization ranges of various environmental parameters. Even with a simplified assumption that all the environmental parameters are uncorrelated, we were able to create a decent curriculum for our robot hand to learn the task.



## Teacher-Guided Curriculum

The idea of *Automatic Curriculum Learning* was proposed by  [Graves, et al. 2017](https://arxiv.org/abs/1704.03003) slightly earlier. It considers a $$$N$-task curriculum as a $$n$$-armed [bandit]({{ site.baseurl }}{% post_url 2018-01-23-the-multi-armed-bandit-problem-and-its-solutions %}) and a syllabus as an adaptive policy which learns to optimize the returns from this bandit. 

Two categories of learning signals have been considered in the paper:
1. Loss-driven progress: the loss function change before and after update. This type of reward signals track the speed of learning process, as the greatest task loss decrease is equivelant to fastest learning.
2. Complex-driven progress: the KL divergence between posterior and prior distribution over network weights. This type of learning signals are inspired by MDL principle, "increasing the model complexity by a certain amount is only worthwhile if it compresses the data by a greater amount". The model complexity is therefore expected to increase most in response to the model nicely generalizing to training examples.

This framework of proposing curriculum automatically through another RL agent was soon formalized as *Teacher-Student Curriculum Learning* (**TSCL**; [Matiisen, et al. 2017](https://arxiv.org/abs/1707.00183)). In TSCL, a *student* is the agent working on actual tasks while a *teacher* agent is the policy . The student aims to master a complex task that might be hard to learn directly. To make this difficult task approachable, we set up another RL agent as the teacher to guide the student agent's learning process by picking propoer subtasks for the student to learn.


![Teacher-student curriculum]({{ '/assets/images/teacher-student-curriculum.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 3. The setup of teacher-student curriculum learning. (Image source: [Matiisen, et al. 2017](https://arxiv.org/abs/1707.00183) + my annotation in red.)*


In the process, the student should learn tasks which: 
1. can help the student make fastest learning progress, or
2. are at risk of being forgotten.

The setup of framing the teacher model as a RL problem is quite similar to Neural Architecture Search (NAS), but differently the RL model in TSCL operates on the task space and NAS operates on the main model architecture space.

Training the teacher model is to solve a [POMDP](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) problem:
- The unobserved $$s_t$$ is the full state of the student model.
- The observed $$o = (x_t^1, \dots, x_t^N)$$ are a list of scores for $$N$$ tasks.
- The action $$a$$ is to pick on subtask.
- The reward per step is the score delta.$$r_t = \sum_{i=1}^N x_t^{(i)} - x_{t-1}^{(i)}$$ (i.e., equivalent to maximizing the score of all tasks at the end of the episode).


The method of estimating learning progress from noisy task scores while balancing exploration vs exploitation can be borrowed from the non-stationary multi-armed bandit problem --- use [$$\epsilon$$-greedy]({{ site.baseurl }}{% post_url 2018-01-23-the-multi-armed-bandit-problem-and-its-solutions %}#ε-greedy-algorithm), or [Thompson sampling]({{ site.baseurl }}{% post_url 2018-01-23-the-multi-armed-bandit-problem-and-its-solutions %}#thompson-sampling).


The core idea, in summary, is to use one policy to propose tasks for another policy to learn better. Interestly, both work above (in discrete task space) found that uniformly sampling from all tasks is a surprisingly strong benchmark.


What if the task space is continuous? [Portelas, et al, 2019](https://arxiv.org/abs/1910.07224) studied on a continuous teacher-student framework, where the teacher instead has to sample parameters from a continuous task sapce to generate a learning curriculum. Given a newly sampled parameter $$p$$, the absolute learning progress (short for ALP) is measured as $$\text{ALP}_p = \vert r - r_\text{old} \vert$$, where $$r$$ is the episodic reward associated with $$p$$ and $r_\text{old}$ is the reward associated with $$p_\text{old}$$, a previous sampled parameter closest to $$p$$ in the task space. This $$p_\text{old}$$ can be retrieved by nearest neighbor. Note that how this ALP score is different from learning signals in TSCL or Grave, et al. 2017 above: ALP score measures the reward difference between two tasks rather than two time steps of the same task.


On top of the task parameter space, [Portelas, et al, 2019](https://arxiv.org/abs/1910.07224)  trains a Gaussian mixture model to fit the distribution of $$\text{ALP}_p$$ over $$p$$. $$\epsilon$$-greedy is used when sampling tasks: with some probability, sample a random task; otherwise sample proportionally to ALP score from the GMM model.



![ALP-GMM]({{ '/assets/images/ALP-GMM-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. The algorithm of ALP-GMM (absolute learning progress Gaussian mixture model). (Image source: [Portelas, et al, 2019](https://arxiv.org/abs/1910.07224))*



## Curriculum through Self-Play

Different from the teacher-student framework, two agents are doing very different things as the teacher learns a different RL task  one is supervising the other. What if we want to make them equal and make both train on the main task directly? How about even make them compete with each other?

[Sukhbaatar, et al. (2017)](https://arxiv.org/abs/1703.05407) proposed a framework for automatic curriculum learning through asymmetric self-play. Two agents, Alice and Bob, play the same task with different goals: Alice challenges Bob to achieve the same state and Bob attempts to complete it as fast as he can.


![Self-play experiments in MazeBase]({{ '/assets/images/self-play-maze.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 5. Illustration of the self-play setup when training two agents. The example task is [MazeBase](https://github.com/facebook/MazeBase): An agent is asked to reach a goal flag in a maze with a light switch, a key and a wall with a door. Toggling the key switch can open or close the door and Turning off the light makes only the glowing light switch available to the agent. (Image source: [Sukhbaatar, et al. 2017](https://arxiv.org/abs/1703.05407))*


Let us consider Alice and Bob as two separate copies for one RL agent trained for the same given environment. Each of them has independent parameters and loss objective. Precisely, training consists of two types of episode:
1. In the *self-play episode*, Alice alters the state from $$s_0$$ to $$s_t$$ and then Bob is asked to return the environment to its original state $$s_0$$ to get an internal reward. 
2. In the *target task episode*, Bob receives an external reward if he visits the flag.


Note that due to the fact that B has to repeat the actions between the same pair of $$(s_0, s_t)$$ of A, this framework only works in reversible or resettable environments.


Alice should learn to push Bob out of comfort zone, but not give him impossible tasks. Bob's reward is set as $$R_B = -\gamma t_B$$ and Alice's reward is $$R_A = \gamma \max(0, t_B - t_A)$$, where $$t_B$$ is the total time for B to complete the task, $$t_A$$ is the time until Alice performs the STOP action and $$\gamma$$ is a scalar constant to rescale the reward to be comparable with the external task reward. If B fails a task, $$t_B = t_\max - t_A$$. 
Both policies are goal-conditioned. The losses imply:
1. B wants to finish a task asap.
2. A prefers tasks that take more time of B.
3. A does not want to take too many steps when B is failing.


In this way, the interaction between Alice and Bob automatically builds a curriculum of increasingly challenging tasks. Meanwhile, as A has done the task herself before proposing the task to B, the task is guaranteed to be solvable.


The paradigm of A suggesting tasks and then B solving them is quite similar to the teacher-student curriculum learning. However, in asymmetric self-play, Alice, who plays a teacher role, also works on the same task to find challenging cases for Bob, rather than optimizes B's learning process explicitly.


## Automatic Goal Generation

Often RL policy need to be able to perform over a set of tasks. The goal should be carefully chosen so that at every training stage, it would not be too hard or too easy for the current policy. A goal $$g \in \mathcal{G}$$ can be defined as a set of states $$S^g$$ and a goal is considered as achieved whenever an agent arrives at any of those states. 

The approach of Generative Goal Learning ([Florensa, et al. 2018](https://arxiv.org/abs/1705.06366)) relies on a Goal GAN to generate desired goals automatically. In their experiment, the reward is very sparse, just a binary flag for whether a goal is achieved or not and the policy is conditioned on goal, 


$$
\begin{aligned}
\pi^{*}(a_t\vert s_t, g) &= \arg\max_\pi \mathbb{E}_{g\sim p_g(.)} R^g(\pi) \\
\text{where }R^g(\pi) &= \mathbb{E}_\pi(.\mid s_t, g) \mathbf{1}[\exists t \in [1,\dots, T]: s_t \in S^g]
\end{aligned}
$$


Here $$R^g(\pi)$$ is the expected return, also equivalent to the success probability. Given  sampled trajectories from the current policy, as long as any state belongs to the goal set, the return will be positive.


Their approach iterates through 3 steps until the policy converges:
1. Label a set of goals based on whether they are at the appropriate level of difficulty for the current policy.
- The set of goals at the appropriate level of difficulty are labeled as **GOID** (short for Goals of Intermediate Difficulty).<br/>$$\text{GOID}_i := \{g : R_\text{min} \leq R^g(\pi_i) \leq R_\text{max} \} \subseteq G$$
2. Train a **Goal GAN** model using labelled goals from step 1 to produce new goals
3. Use these new goals to train the policy, improving its coverage objective.


Here the idea is similar to HER. $R_\text{min}$ and $R_\text{max}$ can be interpreted as a minimum and maximum probability of reaching a goal over T time-steps.

The Goal GAN generates curriculum automatically:
- Generator $$G(z)$$:  produces a new goal and this goal is expected to be a goal uniformly sampled from $$GOID$$ set.
- Discriminator $$D(g)$$: evaluates whether a goal can be achieved. It is expected to tell whether a goal is from $$GOID$$ set.

The Goal GAN is constructed similar to LSGAN ([Least-Squared GAN](https://arxiv.org/abs/1611.04076)) , which has better stability of learning compared to vanilla GAN. According to LSGAN, we should minimize the following losses for $$D$$ and $$G$$ respectively:

$$
\begin{aligned}
\mathcal{L}_\text{LSGAN}(D) &= \frac{1}{2} \mathbb{E}_{g \sim p_\text{data}(g)} [ (D(g) - b)^2] + \frac{1}{2} \mathbb{E}_{z \sim p_z(z)} [ (D(G(z)) - a)^2] \\
\mathcal{L}_\text{LSGAN}(G) &= \frac{1}{2} \mathbb{E}_{z \sim p_z(z)} [ (D(G(z)) - c)^2]
\end{aligned}
$$

where $$a$$ is the label for fake data, $$b$$ for real data, and $$c$$ is the value that $$G$$ wants $$D$$ to believe for fake data. In [Mao et al., (2017)](https://arxiv.org/abs/1611.04076)'s experiments, they used $$a=-1, b=1, c=0$$.

The Goal GAN introduces an extra binary flag $$y_b$$ indicating whether a goal $$g$$ is real ($$y_g = 1$$) or fake ($$y_g = 0$$) so that the model can use negative samples for training:

$$
\begin{aligned}
\mathcal{L}_\text{GoalGAN}(D) &= \frac{1}{2} \mathbb{E}_{g \sim p_\text{data}(g)} [ (D(g) - b)^2 + (1-y_g) (D(g) - a)^2] + \frac{1}{2} \mathbb{E}_{z \sim p_z(z)} [ (D(G(z)) - a)^2] \\
\mathcal{L}_\text{GoalGAN}(G) &= \frac{1}{2} \mathbb{E}_{z \sim p_z(z)} [ (D(G(z)) - c)^2]
\end{aligned}
$$


![Generative goal learning]({{ '/assets/images/generative-goal-learning-algorithm.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 6. The algorithm of Generative Goal Learning. (Image source: ([Florensa, et al. 2018](https://arxiv.org/abs/1705.06366))*


Following the same idea, [Racaniere & Lampinen, et al. (2019)](https://arxiv.org/abs/1909.12892) designs a method to make the objectives of goal generator more sophisticated. Their method contains three components, same as generative goal learning above: 
- **Solver**/Policy $$\pi$$: In each episode, the solver gets a goal $$g$$ at the beginning and get a single binary reward $$R^g$$ at the end.
- **Judge**/Discriminator $$D(.)$$: A classifier to predict the binary reward (goal achieved or not); precisely it outputs the logit of a probability of achieving the given goal, $$\sigma(D(g)) = p(R^g=1\vert g)$$, where $$\sigma$$ is the sigmoid function.
- **Setter**/Generator $$G(.)$$: The goal setter takes as input a desired feasibility score $$f \in \text{Unif}(0, 1)$$ and generates $$g = G(z, f)$$, where the latent variable $$z$$ is sampled by $$z \sim \mathcal{N}(0, I)$$.


The generator is optimized with three objectives:
- (1) Goal **validity**: The proposed goal should be achievable by an expert policy. The corresponding generative loss is designed to increase the likelihood of generating goals that the solver policy has achieved before (like in [HER](https://arxiv.org/abs/1707.01495)).
    - $$\mathcal{L}_\text{val}$$ is the negative log-likelihood of generated goals that have been solved by the solver in the past.
    - Given the goal $$g$$ is sampled as $$g = G(z, f), z \sim \mathcal{N}(0, 1)$$ and $$f$$ is a desired feasibility score, and $$G^{-1}$$ maps backwards from a goal $$g$$ to a latent $$z = G^{-1}(g, f)$$,
    <br/>
    $$
    \mathcal{L}_\text{val} = \mathbb{E}_{\substack{
      g \text{ achieved by solver}, \\
      \xi \in \text{Uniform}(0, \delta), \\
      f \in \text{Uniform}(0, 1)
    }} [-\log p(G^{-1}(g + \xi, f)) ]
    $$

- (2) Goal **feasibility**: The proposed goal should be achievable by the current policy; that is, the level of difficulty should be appropriate.
    - $$\mathcal{L}_\text{feas}$$ is the output probability by the judge model $$D$$ on the generated goal $$G(z, f)$$ should match the desired $f$.
    <br/>
    $$
    \mathcal{L}_\text{feas} = \mathbb{E}_{\substack{
      z \in \mathcal{N}(0, 1), \\
      f \in \text{Uniform}(0, 1)
    }} [D(G(z, f)) - \sigma^{-1}(f)^2 ]
    $$
- (3) Goal **coverage**: We should maximize the entropy of generated goals to encourage diverse goal and to improve the coverage over the goal space.
    - $$\mathcal{L}_\text{cov}$$ is the entropy of $$p(G(z, f))$$,
    <br/>
    $$
    \mathcal{L}_\text{cov} = \mathbb{E}_{\substack{
      z \in \mathcal{N}(0, 1), \\
      f \in \text{Uniform}(0, 1)
    }} [\log p(G(z, f))]
    $$


Their experiments showed complex environment requires all three losses above. When the environment is changing between episodes, both the goal generator and discriminator needs to be conditioned on environmental observation to produce better results. If there is a desired goal distribution, an additional loss can be added to match a desired goal distribution using Wasserstein distance. Using this loss, the generator can push the solver toward mastering the desired tasks more efficiently.


![Goal setter and judge models]({{ '/assets/images/setter-judge-goal-generation.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Training schematic for the (a) solver, (b) judge, and (c) setter models. (Image source: [Racaniere & Lampinen, et al., 2019](https://arxiv.org/abs/1909.12892))*



## Skill-Based Curriculum

Another view is to decompose what an agent can achieve into a variety of skills and naturally each skill set can be mapped into a task. Let's imagine when an agent interacts with the environment in an unsupervised manner, is there a way to discover useful skills from such interaction and further build into the solutions for more complicated tasks?

[Jabri, et al 2019](https://arxiv.org/abs/1912.04226) developed an automatic curriculum, **CARML** (short for "Curricula for Unsupervised Meta-Reinforcement Learning"), by modeling unsupervised trajectories into a latent skill space, with a focus on training [meta-RL]({{ site.baseurl }}{% post_url 2019-06-23-meta-reinforcement-learning %}) policies (i.e. can transfer to unseen tasks). The setting of training environments in CARML are same as in [DIAYN]({{ site.baseurl }}{% post_url 2019-06-23-meta-reinforcement-learning %}#learning-with-random-rewards). An RL algorithm $$f_\theta$$, parameterized by $$\theta$$, is trained via unsupervised interaction formulated as a CMP combined with a learned reward function $$r$$. This setting naturally works for meta-learning purpose, since a customized reward function can be given at the test time.


![CARML]({{ '/assets/images/CARML.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 8. An illustration of CARML, containing two steps: (1) organizing experiential data into the latent skill space; (2) meta-training the policy with the reward function constructed from the learned skills. (Image source: [Jabri, et al 2019](https://arxiv.org/abs/1912.04226))*


CARML is framed as a [variational Expectation-Maximization (EM)](https://chrischoy.github.io/research/Expectation-Maximization-and-Variational-Inference/).

(1) **E-Step**: This is the stage for organizing experiential data. Collected trajectories are modeled with a mixture of latent components forming the [basis](https://en.wikipedia.org/wiki/Basis_(linear_algebra)) of *skills*.

Let $$z$$ be a latent task variable and $$q_\phi$$ be a variational distribution of $$z$$, which could be a mixture model with discrete $$z$$ or a VAE with continuous $$z$$. A variational posterior $$q_\phi(z \vert s)$$ works like a classifier, predicting a skill given a state, and we would like to maximize $$q_\phi(z \vert s)$$ to discriminate between data produced by different skills as much as possible. In E-step, $$q_\phi$$ is fitted to a set of trajectories produced by $$f_\theta$$. 


Precisely, given a trajectory $$\tau = (s_1,\dots,s_T)$$, we would like to find $$\phi$$ such that 

$$
\max_\phi \mathbb{E}_{z\sim q_\phi(z)} \big[ \log q_\phi(\tau \vert z) \big]
= \max_\phi \mathbb{E}_{z\sim q_\phi(z)} \big[ \sum_{s_i \in \tau} \log q_\phi(s_i \vert z) \big]
$$

A simplified assumption is made here to ignore the order of states in one trajectory.

(2) **M-Step**: This is the stage for doing meta-RL training with $$f_\theta$$. The learned skill space is treated as a training task distribution. CARML is agnostic to the type of meta-RL algorithm used in the M-step.

Given a trajectory $$\tau$$, it makes sense for the policy to maximize the mutual information between $$\tau$$ and $$z$$, $$I(\tau;z) = H(\tau) - H(\tau \vert z)$$, because:
maximizing $$H(\tau)$$ => diversity in the policy data space; expected to be large.
minimizing $$H(\tau \vert z)$$ => given a certain skill, the behavior should be restricted; expected to be small.

Then we have,

$$
\begin{aligned}
I(\tau; z) 
&= \mathcal{H}(z) - \mathcal{H}(z \vert s_1,\dots, s_T) \\
&\geq \mathbb{E}_{s \in \tau} [\mathcal{H}(z) - \mathcal{H}(z\vert s)] & \text{\small{; discard the order of states.}}\\
&= \mathbb{E}_{s \in \tau} [\mathcal{H}(s_t) - \mathcal{H}(s\vert z)] & \text{\small{; by definition of MI.}} \\
&= \mathbb{E}_{z\sim q_\phi(z), s\sim \pi_\theta(s|z)} [\log q_\phi(s|z) - \log \pi_\theta(s)] \\
&\approx \mathbb{E}_{z\sim q_\phi(z), s\sim \pi_\theta(s|z)} [\color{red}{\log q_\phi(s|z) - \log q_\phi(s)}] & \text{\small{; assume learned marginal distr. matches policy.}}
\end{aligned}
$$

We can set the reward as $$\log q_\phi(s|z) - \log q_\phi(s)$$, as shown in the red part above. In order to balance between task-specific exploration (as in red below) and latent skill matching (as in blue below) , a parameter $$\lambda \in [0, 1]$$ is added. Each realization of $$z \sim q_\phi(z)$$ induces a reward function $$r_z(s)$$ (remember that reward + CMP => MDP) as follows:


$$
\begin{aligned}
r_z(s)
&= \lambda \log q_\phi(s|z) - \log q_\phi(s) \\
&= \lambda \log q_\phi(s|z) - \log \frac{q_\phi(s|z) q_\phi(z)}{q_\phi(z|s)} \\
&= \lambda \log q_\phi(s|z) - \log q_\phi(s|z) - \log q_\phi(z) + \log q_\phi(z|s) \\
&= (\lambda - 1) \log \color{red}{q_\phi(s|z)} + \color{blue}{\log q_\phi(z|s)} + C
\end{aligned}
$$


![CARML algorithm]({{ '/assets/images/CARML-algorithm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 9. The algorithm of CARML. (Image source: [Jabri, et al 2019](https://arxiv.org/abs/1912.04226))*


Learning latent skill space can be done in a different way such as in [Hausman, et al. 2018](https://openreview.net/forum?id=rk07ZXZRb). The goal is to learn a task-conditioned policy, $$\pi(a|s, t)$$, where $$t$$ is from a discrete list of tasks, $$t \in \mathcal{T} = [t_1, \dots, t_T]$$. However, rather than learning $$T$$ separate solutions, one per task, it would be nice to learn a latent skill space so that each task could be represented in a distribution over skills and thus skills are reused between tasks.


The policy would be defined as $$\pi_\theta(a|s,t) = \int \pi_\theta(a \vert z,s,t) p_\phi(z|t)\mathrm{d}z$$, where $$\pi_\theta$$ and $$p_\phi$$ are policy and embedding networks to learn, respectively.  If $$z$$ is discrete, i.e. drawn from a set of $$K$$ skills, then the policy becomes a mixture of $$K$$ sub-policies. The policy training uses [SAC](http://127.0.0.1:4000/lil-log/2018/04/07/policy-gradient-algorithms.html#sac) and the dependency on $$z$$ is introduced in the entropy term.



## References

[1] Jeffrey L. Elman. ["Learning and development in neural networks: The importance of starting small."](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.128.4487&rep=rep1&type=pdf) Cognition 48.1 (1993): 71-99.

[2] Yoshua Bengio, et al. ["Curriculum learning."](https://www.researchgate.net/profile/Y_Bengio/publication/221344862_Curriculum_learning/links/546cd2570cf2193b94c577ac/Curriculum-learning.pdf) ICML 2009.

[3] Daphna Weinshall, Gad Cohen, and Dan Amir. ["Curriculum learning by transfer learning: Theory and experiments with deep networks."](https://arxiv.org/abs/1802.03796) ICML 2018.

[4] Wojciech Zaremba and Ilya Sutskever. ["Learning to execute."](https://arxiv.org/abs/1410.4615) arXiv preprint arXiv:1410.4615 (2014).

[5] Tambet Matiisen, et al. ["Teacher-student curriculum learning."](https://arxiv.org/abs/1707.00183) IEEE Trans. on neural networks and learning systems (2017).

[6] Alex Graves, et al. ["Automated curriculum learning for neural networks."](https://arxiv.org/abs/1704.03003) ICML 2017.

[7]  Remy Portelas, et al. [Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments](https://arxiv.org/abs/1910.07224). CoRL 2019.

[8] Sainbayar Sukhbaatar, et al. ["Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play."](https://arxiv.org/abs/1703.05407) ICLR 2018.

[9] Carlos Florensa, et al. ["Automatic Goal Generation for Reinforcement Learning Agents"](https://arxiv.org/abs/1705.06366) ICML 2019.

[10] Sebastien Racaniere & Andrew K. Lampinen, et al. ["Automated Curriculum through Setter-Solver Interactions"](https://arxiv.org/abs/1909.12892) ICLR 2020.

[11] Allan Jabri, et al. ["Unsupervised Curricula for Visual Meta-Reinforcement Learning"](https://arxiv.org/abs/1912.04226) NeuriPS 2019.

[12] Karol Hausman, et al. ["Learning an Embedding Space for Transferable Robot Skills "](https://openreview.net/forum?id=rk07ZXZRb) ICLR 2018.

[13] Josh Merel, et al. ["Reusable neural skill embeddings for vision-guided whole body movement and object manipulation"](https://arxiv.org/abs/1911.06636) arXiv preprint arXiv:1911.06636 (2019).

[14] OpenAI, et al. ["Solving Rubik's Cube with a Robot Hand."](https://arxiv.org/abs/1910.07113) arXiv preprint arXiv:1910.07113 (2019).

