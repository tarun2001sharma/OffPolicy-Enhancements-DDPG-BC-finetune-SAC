# Off-Policy Reinforcement Learning Enhancements: A Comprehensive Study

## Abstract

This repository presents a comprehensive study of off-policy reinforcement learning (RL) techniques implemented using an actor-critic framework. We explore several approaches: a baseline DDPG-like algorithm for RL training, an integration of behavior cloning (BC) with online RL fine-tuning to improve sample efficiency, a discussion of advanced RL variants (such as Double Q-learning and Dueling DQN), and a bonus implementation of Soft Actor-Critic (SAC) that learns its exploration noise online. Detailed theoretical insights and mathematical formulations are provided, along with experimental analyses using training and evaluation curves.

## Introduction

Reinforcement learning (RL) aims to learn policies that maximize cumulative reward by interacting with an environment. Off-policy methods, in particular, use data generated from previous interactions (or even from another policy) to update the current policy. In this work, we focus on an actor-critic framework, where two neural networks are trained: one to approximate the state-action value function (critic) and another to represent the policy (actor).

The baseline method is inspired by the Deep Deterministic Policy Gradients (DDPG) algorithm, which learns a deterministic policy using gradients computed from the critic. We then enhance this framework by pre-training the actor through behavior cloning (BC) using expert demonstrations, followed by RL fine-tuning. Finally, we extend our exploration by implementing Soft Actor-Critic (SAC), which learns an adaptive exploration noise via entropy regularization.

## Methodology

### Baseline RL Training (DDPG-like Algorithm)

In the baseline approach, the RL agent learns a deterministic policy via an actor-critic framework. The critic estimates the Q-function, defined as

$$
Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r_t \,\middle|\, s_0 = s,\, a_0 = a\right],
$$

and is updated by minimizing the temporal difference (TD) error:

$$
L_{\text{critic}} = \mathbb{E}\left[\left(Q(s,a) - \Big(r + \gamma\, Q_{\text{target}}(s', a')\Big)\right)^2\right].
$$

The actor is trained to maximize the expected Q-value, which is equivalent to minimizing

$$
L_{\text{actor}} = -\mathbb{E}\left[ Q\big(s,\pi(s)\big) \right].
$$

A target network is maintained to provide a stable target for the critic. The target network parameters, $\theta_{\text{target}}$, are updated softly:

$$
\theta_{\text{target}} \leftarrow \tau\, \theta + (1-\tau)\, \theta_{\text{target}}.
$$

### Behavior Cloning (BC) Training and RL Fine-Tuning

To enhance sample efficiency, the agent is initially pre-trained using behavior cloning (BC) on expert demonstration data. The BC loss is defined as:

$$
L_{\text{BC}} = \mathbb{E}\left[\|\pi(s) - a_{\text{expert}}\|^2\right].
$$

After pre-training, the actor is fine-tuned using RL updates. The overall actor loss during fine-tuning is a combination of the RL objective and the BC loss:

$$
L_{\text{actor}} = L_{\text{RL}} + \alpha\, L_{\text{BC}} = -\mathbb{E}\left[ Q\big(s,\pi(s)\big) \right] + \alpha\, \mathbb{E}\left[\|\pi(s) - a_{\text{expert}}\|^2\right],
$$

where $\alpha$ is a hyperparameter that balances reward maximization against imitation of the expert policy.

### RL Variants

#### Double Q-Learning

Double Q-learning reduces the overestimation bias found in conventional Q-learning by decoupling action selection and evaluation. Its target is computed as:

$$
y = r + \gamma\, Q_{\text{target}}\Big(s',\, \arg\max_{a'} Q(s',a';\theta_1);\theta_2\Big).
$$

#### Dueling DQN

Dueling DQN decomposes the Q-value into a state-value function and an advantage function:

$$
Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a') \right).
$$

These techniques, though originally designed for discrete-action spaces, can inspire similar modifications in actor-critic architectures (e.g., using multiple critics or decomposing value functions) to stabilize training and reduce estimation bias.

### Soft Actor-Critic (SAC) Implementation

SAC is an off-policy algorithm that incorporates maximum entropy reinforcement learning, leading to robust exploration and improved stability. Key modifications for SAC include:

1. **Actor Network Changes:**  
   The actor now outputs both a mean $\mu(s)$ and a log standard deviation $\log \sigma(s)$, defining a Gaussian policy:

   $$
   \pi(s) \sim \mathcal{N}\big(\mu(s),\, \sigma(s)^2\big).
   $$

2. **Entropy Regularization:**  
   The actor loss includes an entropy bonus to encourage exploration:

   $$
   L_{\text{actor}} = \mathbb{E}\left[\alpha\, \log \pi(a|s) - Q(s,a)\right].
   $$

3. **Temperature Parameter:**  
   A learnable temperature parameter $\alpha$ balances the trade-off between reward maximization and entropy. An auxiliary loss for tuning $\alpha$ is:

   $$
   L(\alpha) = -\alpha\, \mathbb{E}\left[\log \pi(a|s) + \mathcal{H}_{\text{target}}\right].
   $$

4. **Critic Target Update with Entropy:**  
   The critic target incorporates the entropy term:

   $$
   y = r + \gamma \left( Q_{\text{target}}(s',a') - \alpha\, \log \pi(a'|s') \right).
   $$

## Experiments and Evaluation

### Metrics and Plots

To assess performance, the following plots are essential:

- **Training Curve:**  
  $$x: \text{Frames (or Steps)} \quad y: \text{Episode Reward}$$
  
  This plot illustrates how the agent's performance evolves as it interacts with the environment.

- **Evaluation Curve:**  
  $$x: \text{Frames (or Steps)} \quad y: \text{Episode Reward (evaluation mode)}$$
  
  Evaluation curves typically use a deterministic policy (e.g., mean action) and reveal the true performance of the learned policy.

- **BC Loss Curve (Optional for Q2):**  
  $$x: \text{Iterations} \quad y: L_{\text{BC}}$$
  
  This plot shows the convergence of the behavior cloning loss during pre-training.

- **Temperature Parameter Plot (Optional for SAC):**  
  $$x: \text{Frames (or Steps)} \quad y: \alpha$$
  
  For SAC, monitoring the evolution of the temperature parameter provides insight into the balance between exploration and exploitation.

### Experimental Setup

- **Environment:** The goal-reaching environment modified from Assignment 1.
- **Baselines:**  
  - RL Training Only (DDPG-style)
  - BC Pre-training + RL Fine-Tuning
  - SAC (Bonus)
- **Analysis:** Compare the sample efficiency, convergence speed, and stability of each method based on the training and evaluation curves.

## Conclusion

This work explores the integration of multiple off-policy reinforcement learning techniques to improve performance and sample efficiency. The baseline actor-critic method provides a foundation, while behavior cloning pre-training enhances early performance, and SAC introduces an adaptive mechanism for exploration. We also discuss potential improvements from advanced RL variants such as Double Q-learning and Dueling DQN.

The experiments demonstrate:
- Faster convergence and higher initial performance with BC pre-training.
- Improved stability and robustness with SAC due to entropy regularization.
- The potential benefits of using multiple critics and decomposed value functions to mitigate overestimation bias.

These insights are discussed in detail in the accompanying PDF submission.

## Requirements and Setup

### Requirements
- Python 3.x  
- $$\texttt{numpy},\ \texttt{scipy},\ \texttt{torch}$$  
- $$\texttt{pandas},\ \texttt{matplotlib}$$  
- $$\texttt{hydra-core}$$ (for configuration management)  
- $$\texttt{particle-envs}$$ (for the environment)

### Setup

Create the Conda Environment:**
   ```bash
   conda env create -f conda_env.yml
   conda activate ddrl
```

### Dependencies

- Python 3.x, NumPy, SciPy, PyTorch
- Pandas, Matplotlib, Hydra-core
- Particle-envs

### Installation

```bash
conda env create -f conda_env.yml
conda activate ddrl
pip install -e particle-envs/
```

### Run Experiments

- RL Only: `python train_rl.py`
- BC+RL: `python train_bcrl.py`
- SAC (Bonus): `python train_sac.py`

## References

- Lillicrap et al. (2015), "Continuous control with deep reinforcement learning."
- Haarnoja et al. (2018), "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning."
- Mnih et al. (2015), "Human-level control through deep reinforcement learning."
