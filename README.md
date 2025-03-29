# Off-Policy Reinforcement Learning Enhancements

This repository contains implementations and experiments for an assignment in **Deep Decision Making and Reinforcement Learning (CSCI-GA 3033-090)**. The focus is on off-policy RL methods using an actor-critic framework. In this project, we:
- Implement a DDPG-like algorithm for **RL Training Only**.
- Combine **Behavior Cloning (BC)** with RL fine-tuning to enhance sample efficiency.
- Discuss RL variants such as Double Q-learning and Dueling DQN.
- Implement a bonus **Soft Actor-Critic (SAC)** agent that learns its exploration noise online.

---

## Environment and Setup

The environment is a modified goal-reaching task conforming to the OpenAI Gym API. Key changes include a fixed episode length (50) and modified reward functions. The repository is structured to work with the provided `conda_env.yml` and the modified `particle-envs`.

---

## Q1: RL Training Only

In the first part, we implement a basic off-policy actor-critic method inspired by DDPG.

### Theoretical Background

The **Q-function** is defined as:

$$
Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r_t \,\middle|\, s_0 = s,\ a_0 = a\right]
$$

During training, the critic is updated to minimize the Temporal Difference (TD) error:

$$
L_{\text{critic}} = \mathbb{E}\left[ \left(Q(s,a) - \left(r + \gamma\, Q_{\text{target}}(s',a')\right)\right)^2 \right]
$$

The actor is updated to maximize the Q-value, which is equivalent to minimizing:

$$
L_{\text{actor}} = -\mathbb{E}\left[ Q(s,\pi(s)) \right]
$$

A **target network** is maintained for the critic to provide a stable target:

$$
Q_{\text{target}}(s', a') \quad \text{with soft updates:} \quad \theta_{\text{target}} \leftarrow \tau\, \theta + (1-\tau)\, \theta_{\text{target}}
$$

### Plots to Include

- **Training Curve:**  
  $ x: \text{Frames (or Steps)} \quad y: \text{Episode Reward} $
  
- **Evaluation Curve:**  
  $$ x: \text{Frames (or Steps)} \quad y: \text{Episode Reward (evaluation mode)} $$

---

## Q2: BC Training and RL Fine-Tuning

To enhance sample efficiency, the policy is first pre-trained using behavior cloning (BC) and then fine-tuned with RL.

### Theoretical Background

The **Behavior Cloning (BC) loss** is:

$$
L_{\text{BC}} = \mathbb{E}\left[ \|\pi(s) - a_{\text{expert}}\|^2 \right]
$$

During fine-tuning, the actor loss becomes a combination of the RL objective and the BC loss:

$$
L_{\text{actor}} = L_{\text{RL}} + \alpha\, L_{\text{BC}} = -\mathbb{E}\left[ Q(s,\pi(s)) \right] + \alpha\, \mathbb{E}\left[ \|\pi(s) - a_{\text{expert}}\|^2 \right]
$$

where $$ \alpha $$ is a hyperparameter balancing the two objectives.

### Plots to Include

- **Evaluation Curve Comparison (RL vs. BC+RL):**  
  $$ x: \text{Frames (or Steps)} \quad y: \text{Episode Reward (evaluation mode)} $$
  
- **(Optional) BC Loss Curve:**  
  $$ x: \text{Iterations} \quad y: L_{\text{BC}} $$

---

## Q3: RL Variants

### Discussion of RL Variants

#### Double Q-learning

Double Q-learning addresses the overestimation bias by decoupling action selection from action evaluation. The target is computed as:

$$
y = r + \gamma\, Q_{\text{target}}\Big(s',\, \arg\max_{a'} Q(s',a';\theta_1);\theta_2\Big)
$$

#### Dueling DQN

Dueling DQN decomposes the Q-value into a state-value function and an advantage function:

$$
Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)
$$

#### Application to Actor-Critic

While these variants were originally designed for discrete-action settings (DQN), similar ideas (e.g., using two critics) can potentially stabilize training in actor-critic frameworks by reducing overestimation bias and better isolating state values.

### Plots (if Experimented)

- **Standard Reward vs. Frames** for any modified architecture.  
  $$ x: \text{Frames} \quad y: \text{Episode Reward} $$

---

## Bonus: Soft Actor-Critic (SAC) Implementation

The bonus task is to implement SAC, which learns the standard deviation (i.e., exploration noise) and incorporates an entropy bonus.

### Key Modifications for SAC

1. **Actor Network Changes:**

   The actor now outputs both a mean $$ \mu(s) $$ and a log standard deviation $$ \log \sigma(s) $$:

   $$
   \pi(s) \sim \mathcal{N}\big(\mu(s),\, \sigma(s)^2\big)
   $$

2. **Entropy Regularization in Actor Loss:**

   The actor loss becomes:

   $$
   L_{\text{actor}} = \mathbb{E}\left[\alpha\, \log \pi(a|s) - Q(s,a)\right]
   $$

3. **Temperature Parameter:**

   Introduce a temperature $$ \alpha $$ (learned via:

   $$
   L(\alpha) = -\alpha\, \mathbb{E}\left[\log \pi(a|s) + \mathcal{H}_{\text{target}}\right]
   $$

   ) which balances reward maximization and entropy (exploration).

4. **Critic Target Update with Entropy Term:**

   The target for the critic is now computed as:

   $$
   y = r + \gamma \Big( Q_{\text{target}}(s', a') - \alpha\, \log \pi(a'|s') \Big)
   $$

### Plots to Include

- **Training and Evaluation Curves:**  
  $$ x: \text{Frames} \quad y: \text{Episode Reward} $$  
  Compare SAC against the baseline RL.
  
- **(Optional) Temperature Parameter Plot:**  
  $$ x: \text{Frames} \quad y: \alpha $$

---

## Requirements and Running the Code

### Requirements
- Python 3.x
- `numpy`, `scipy`, `torch`
- `pandas`, `matplotlib`
- `hydra-core` (for configuration management)
- `particle-envs` (for the environment)

### Setup
1. Create the conda environment:
   ```bash
   conda env create -f conda_env.yml
   conda activate ddrl
