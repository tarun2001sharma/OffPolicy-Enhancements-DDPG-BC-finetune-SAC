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

In the first part, we implement a basic off-policy actor-critic method inspired by DDPG. The key idea is to learn the **Q-value function** and a deterministic policy using deep networks.

### Theoretical Background

The **Q-function** is defined as:

$$
Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s,\ a_0 = a\right]
$$

During training, the critic is updated to minimize the Temporal Difference (TD) error:

$$
L_{\text{critic}} = \mathbb{E}\Big[ \big(Q(s,a) - \Big(r + \gamma\, Q_{\text{target}}(s',a')\Big)\big)^2\Big]
$$

The actor is updated to maximize the Q-value:

$$
L_{\text{actor}} = -\mathbb{E}[Q(s,\pi(s))]
$$

where the target network $Q_{\text{target}}$ is a slow-moving copy of the critic to improve stability.

### Plots to Include

- **Training Curve:**  
  $$ x: \text{Frame (or Steps)} \quad y: \text{Episode Reward} $$  
- **Evaluation Curve:**  
  $$ x: \text{Frame (or Steps)} \quad y: \text{Episode Reward (evaluation mode)} $$

---

## Q2: Behavior Cloning (BC) Training and RL Fine-Tuning

To boost sample efficiency, we pre-train the actor using behavior cloning from expert data (provided in `bc.pkl`), and then fine-tune it with RL.

### Theoretical Background

**Behavior Cloning (BC)** is a supervised learning approach. The BC loss is given by:

$$
L_{\text{BC}} = \mathbb{E}\Big[\|\pi(s) - a_{\text{expert}}\|^2\Big]
$$

The combined actor loss during fine-tuning becomes:

$$
L_{\text{actor}} = L_{\text{RL}} + \alpha\, L_{\text{BC}} = -\mathbb{E}[Q(s,\pi(s))] + \alpha\, \mathbb{E}\Big[\|\pi(s) - a_{\text{expert}}\|^2\Big]
$$

where $$ \alpha $$ is a hyperparameter balancing imitation and reward maximization.

### Plots to Include

- **Evaluation Curve Comparison:**  
  $$ x: \text{Frame (or Steps)} \quad y: \text{Episode Reward (evaluation mode)} $$  
  Compare the performance of RL-only vs. BC+RL (BCRL).
- **Optional:** A plot of BC loss vs. training iterations to show how quickly the actor learns from the expert.

---

## Q3: RL Variants

### Discussion

- **Double Q-Learning:**  
  This method uses two Q-networks to reduce overestimation bias by decoupling the action selection and evaluation:
  
  $$ 
  y = r + \gamma\, Q_{\text{target}}(s', \arg\max_{a'} Q(s',a';\theta_1); \theta_2)
  $$

- **Dueling DQN:**  
  In dueling architectures, the Q-value is decomposed into a state-value function and an advantage function:
  
  $$
  Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a') \right)
  $$

### Application in Actor-Critic

While these methods were initially proposed for DQN, similar ideas (e.g., using two critics) may help stabilize actor-critic methods. However, their integration is non-trivial due to the continuous action space.

---

## Bonus: Soft Actor-Critic (SAC) Implementation

The bonus task is to implement **SAC**, which replaces a fixed exploration standard deviation with a learned one and introduces an entropy term in the actor’s objective.

### Key Modifications for SAC

1. **Actor Network Output:**  
   Instead of a fixed standard deviation, the actor outputs both a mean $$ \mu(s) $$ and a log standard deviation $$ \log \sigma(s) $$:

   $$
   \pi(s) \sim \mathcal{N}\big(\mu(s),\, \sigma(s)^2\big)
   $$

2. **Entropy Regularization:**  
   The actor loss is augmented by an entropy bonus:

   $$
   L_{\text{actor}} = \mathbb{E}\Big[\alpha\, \log \pi(a|s) - Q(s,a)\Big]
   $$

3. **Temperature Parameter:**  
   Introduce a learnable temperature $$ \alpha $$ (or keep it fixed), with optional automatic tuning. One common approach is to minimize:

   $$
   L(\alpha) = -\alpha\, \mathbb{E}\Big[\log \pi(a|s) + \mathcal{H}_{\text{target}}\Big]
   $$

4. **Critic Update:**  
   The target for the critic includes the entropy term:

   $$
   y = r + \gamma \Big(Q_{\text{target}}(s', a') - \alpha\, \log \pi(a'|s')\Big)
   $$

### Plots to Include

- **Training & Evaluation Curves:**  
  $$ x: \text{Frame (or Steps)} \quad y: \text{Episode Reward} $$
- **Optional:** Temperature parameter $$ \alpha $$ over time:
  
  $$ x: \text{Frame (or Steps)} \quad y: \alpha $$

---

## How to Run

1. **Setup Environment:**  
   Use the provided `conda_env.yml` to create and activate your environment.
   
   ```bash
   conda env create -f conda_env.yml
   conda activate ddrl
