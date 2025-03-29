
# Off-Policy Reinforcement Learning Enhancements: A Comprehensive Study

## Abstract

This repository presents a comprehensive study of off-policy reinforcement learning (RL) using an actor-critic framework, covering baseline DDPG-like training, behavior cloning (BC) pre-training with RL fine-tuning, exploration of advanced RL variants, and a bonus implementation of Soft Actor-Critic (SAC). Theoretical foundations and mathematical formulations are discussed along with experimental analyses.

## Introduction

Off-policy RL methods use data from previous interactions or other policies to update the current policy. We use an actor-critic approach, involving two neural networks: the critic (estimating Q-values) and the actor (policy network).

## Methodology

### Baseline RL Training (DDPG-like Algorithm)

The critic estimates the Q-function:
$$
Q(s, a) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \middle| s_0 = s, a_0 = a 
ight]
$$

Critic updates minimize TD-error:
$$
L_{	ext{critic}} = \mathbb{E}\left[(Q(s,a)-(r+\gamma Q_{	ext{target}}(s',a')))^2
ight]
$$

Actor updates maximize the critic's Q-values:
$$
L_{	ext{actor}} = -\mathbb{E}[Q(s,\pi(s))]
$$

### BC Training and RL Fine-Tuning

Initial training uses behavior cloning (BC):
$$
L_{	ext{BC}} = \mathbb{E}[\|\pi(s) - a_{	ext{expert}}\|^2]
$$

Fine-tuning combines RL and BC objectives:
$$
L_{	ext{actor}} = -\mathbb{E}[Q(s,\pi(s))] + lpha \mathbb{E}[\|\pi(s) - a_{	ext{expert}}\|^2]
$$

### RL Variants

- **Double Q-Learning:** Reduces overestimation:
$$
y = r + \gamma Q_{	ext{target}}(s', rg\max_{a'} Q(s',a';	heta_1);	heta_2)
$$

- **Dueling DQN:** Decomposes Q-values:
$$
Q(s,a) = V(s) + \left(A(s,a) - rac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')
ight)
$$

### Soft Actor-Critic (SAC)

SAC uses a stochastic actor with entropy regularization. The actor outputs a mean and log-standard deviation:
$$
\pi(s) \sim \mathcal{N}(\mu(s), \sigma(s)^2)
$$

Actor loss with entropy:
$$
L_{	ext{actor}} = \mathbb{E}[lpha\log\pi(a|s)-Q(s,a)]
$$

Temperature parameter ($lpha$) is optimized by:
$$
L(lpha) = -lpha \mathbb{E}[\log\pi(a|s)+\mathcal{H}_{	ext{target}}]
$$

Critic update includes entropy:
$$
y = r + \gamma(Q_{	ext{target}}(s',a')-lpha\log\pi(a'|s'))
$$

## Experiments

- **Training and Evaluation Curves** (Frames vs Episode Reward)
- **BC Loss Curve** (optional for BC)
- **Temperature ($lpha$) Plot** (optional for SAC)

## Setup

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
