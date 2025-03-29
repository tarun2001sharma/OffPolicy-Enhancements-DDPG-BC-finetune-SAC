# Off-Policy Reinforcement Learning Enhancements

This repository contains implementations and experiments for an assignment in **Deep Decision Making and Reinforcement Learning (CSCI-GA 3033-090)**. The focus is on off-policy reinforcement learning using an actor-critic framework. In this project, we:

- Implement a DDPG-like algorithm for **RL Training Only**.
- Combine **Behavior Cloning (BC)** with RL fine-tuning to improve sample efficiency.
- Discuss RL variants such as **Double Q-learning** and **Dueling DQN**.
- Implement a bonus **Soft Actor-Critic (SAC)** agent that learns its exploration noise online.

---

## Environment and Setup

The environment is a modified goal-reaching task conforming to the OpenAI Gym API. Key modifications include a fixed episode length (50) and modified reward functions. The repository is structured to work with the provided `conda_env.yml` and the modified `particle-envs`.

---

## Q1: RL Training Only

### Theoretical Background

The **Q-function** is defined as

$$
Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \, r_t \,\middle|\, s_0 = s,\, a_0 = a\right]
$$

During training, the critic is updated to minimize the temporal difference (TD) error:

$$
L_{\text{critic}} = \mathbb{E}\left[\left(Q(s,a) - \Big(r + \gamma\, Q_{\text{target}}(s', a')\Big)\right)^2\right]
$$

The actor is updated to maximize the Q-value, which is equivalent to minimizing:

$$
L_{\text{actor}} = -\mathbb{E}\left[ Q\big(s,\pi(s)\big) \right]
$$

A target network is maintained for the critic to provide a stable target, updated with soft updates:

$$
\theta_{\text{target}} \leftarrow \tau\, \theta + (1-\tau)\, \theta_{\text{target}}
$$

### Key Plots

- **Training Curve:**  
  $$ x: \text{Frames (or Steps)} \quad y: \text{Episode Reward} $$

- **Evaluation Curve:**  
  $$ x: \text{Frames (or Steps)} \quad y: \text{Episode Reward (evaluation mode)} $$

---

## Q2: Behavior Cloning (BC) Training and RL Fine-Tuning

### Theoretical Background

To improve sample efficiency, the policy is first pre-trained using behavior cloning. The **Behavior Cloning loss** is given by:

$$
L_{\text{BC}} = \mathbb{E}\left[\|\pi(s) - a_{\text{expert}}\|^2\right]
$$

During fine-tuning, the actor loss becomes a combination of the RL objective and the BC loss:

$$
L_{\text{actor}} = L_{\text{RL}} + \alpha\, L_{\text{BC}} = -\mathbb{E}\left[ Q\big(s,\pi(s)\big) \right] + \alpha\, \mathbb{E}\left[\|\pi(s) - a_{\text{expert}}\|^2\right]
$$

where $$ \alpha $$ is a hyperparameter that balances the two terms.

### Key Plots

- **Evaluation Curve Comparison (RL vs. BC+RL):**  
  $$ x: \text{Frames (or Steps)} \quad y: \text{Episode Reward (evaluation mode)} $$

- **(Optional) BC Loss Curve:**  
  $$ x: \text{Iterations} \quad y: L_{\text{BC}} $$

---

## Q3: RL Variants

### Discussion

#### Double Q-Learning

Double Q-learning addresses overestimation bias by decoupling the action selection and evaluation. The target is computed as:

$$
y = r + \gamma\, Q_{\text{target}}\Big(s',\, \arg\max_{a'} Q(s',a';\theta_1);\theta_2\Big)
$$

#### Dueling DQN

Dueling DQN decomposes the Q-value into a state-value function and an advantage function:

$$
Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right)
$$

#### Application to Actor-Critic

Although these variants were originally designed for discrete-action settings (e.g., DQN), similar ideas—such as using multiple critics or decomposing value functions—can potentially stabilize training in actor-critic frameworks by reducing overestimation bias and isolating state values.

### (Optional) Plot

If you experiment with these variants, you can plot:

$$
x: \text{Frames} \quad y: \text{Episode Reward}
$$

to compare modified architectures against the baseline.

---

## Bonus: Soft Actor-Critic (SAC) Implementation

### Key Modifications for SAC

1. **Actor Network Changes:**

   The actor now outputs both a mean $$ \mu(s) $$ and a log standard deviation $$ \log \sigma(s) $$, forming a Gaussian policy:

   $$
   \pi(s) \sim \mathcal{N}\big(\mu(s),\, \sigma(s)^2\big)
   $$

2. **Entropy Regularization:**

   The actor loss is modified to include an entropy bonus:

   $$
   L_{\text{actor}} = \mathbb{E}\left[\alpha\, \log \pi(a|s) - Q(s,a)\right]
   $$

3. **Temperature Parameter:**

   Introduce a learnable temperature $$ \alpha $$ that balances reward maximization and exploration. An optional loss for tuning $$ \alpha $$ is:

   $$
   L(\alpha) = -\alpha\, \mathbb{E}\left[\log \pi(a|s) + \mathcal{H}_{\text{target}}\right]
   $$

4. **Critic Target Update with Entropy:**

   The critic target now incorporates the entropy term:

   $$
   y = r + \gamma \left( Q_{\text{target}}(s',a') - \alpha\, \log \pi(a'|s') \right)
   $$

### Key Plots

- **Training & Evaluation Curves:**  
  $$ x: \text{Frames} \quad y: \text{Episode Reward} $$  
  Compare SAC performance with the baseline RL.

- **(Optional) Temperature Parameter Plot:**  
  $$ x: \text{Frames} \quad y: \alpha $$

---

## Requirements and Running the Code

### Requirements

- Python 3.x  
- $$\texttt{numpy},\ \texttt{scipy},\ \texttt{torch}$$  
- $$\texttt{pandas},\ \texttt{matplotlib}$$  
- $$\texttt{hydra-core}$$ (for configuration management)  
- $$\texttt{particle-envs}$$ (for the environment)

### Setup

1. **Create the Conda Environment:**

   ```bash
   conda env create -f conda_env.yml
   conda activate ddrl
