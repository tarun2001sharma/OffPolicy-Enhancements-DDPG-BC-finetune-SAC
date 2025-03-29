# OffPolicy-Enhancements-DDPG-BC-finetune-SAC
A repository implementing off-policy reinforcement learning techniques including DDPG, behavior cloning with RL fine-tuning, and SAC.

# DeepRL-OffPolicy

## Overview

This repository implements off-policy reinforcement learning (RL) algorithms with several enhancements:
- **RL Training Only** using a DDPG-like actor-critic approach.
- **Behavior Cloning (BC) Pre-training and RL Fine-Tuning** to improve sample efficiency.
- **RL Variants Discussion** (Double Q-learning and Dueling DQN) in the context of actor-critic methods.
- **Bonus: Soft Actor-Critic (SAC) Implementation** that learns its exploration standard deviation and includes an entropy bonus.

The environment is a goal-reaching task (a modified version of the Assignment 1 environment), and the code conforms to the OpenAI Gym API.

## Assignment Details

### Q1: RL Training Only

- **Task:** Complete the TODO sections in the files within `policy/agent/` for the actor, critic, and RL agent.
- **Objective:** Learn an off-policy RL algorithm based on actor-critic methods (inspired by DDPG).
- **Key Metrics:** Training and evaluation curves showing episode reward vs. frames/steps.

#### Theoretical Background

The Q-value function is defined as:
$
Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]
$
This function represents the expected cumulative (discounted) reward when starting in state $$ s $$, taking action $$ a $$, and following the current policy thereafter.

The critic is updated using a loss function that minimizes the difference between its predicted Q-value and a target value computed by:
$$
\text{target } Q = r + \gamma \, Q_{\text{target}}(s', a')
$$
The actor is updated by maximizing the expected Q-value:
$$
L_{\text{actor}} = -\mathbb{E}\left[Q(s, a)\right]
$$

### Q2: BC Training and RL Fine-Tuning

- **Task:** Complete the TODOs in `policy/agent/bcrl.py` where the actor loss is a combination of the RL loss and a behavior cloning (BC) loss.
- **Objective:** Pre-train the actor with expert demonstrations using BC loss:
$$
L_{\text{BC}} = \mathbb{E}\left[\| \pi(s) - a_{\text{expert}} \|^2\right]
$$
Then, fine-tune the policy online with RL where the overall actor loss becomes:
$$
L_{\text{actor}} = L_{\text{RL}} + \alpha \, L_{\text{BC}}
$$
with $$ L_{\text{RL}} = -\mathbb{E}\left[Q(s, a)\right] $$ and $$ \alpha $$ being the BC weight.

- **Key Metrics:** Compare training and evaluation curves (episode reward vs. frames/steps) for RL-only and BC+RL approaches.

### Q3: RL Variants

- **Task:** Briefly describe how Double Q-learning and Dueling DQN differ from the provided code.
- **Objective:** Explain whether these methods could improve performance in an actor-critic based off-policy RL algorithm.

#### Discussion Points

- **Double Q-learning** reduces overestimation bias by decoupling action selection from action evaluation.
- **Dueling DQN** decomposes the Q-function into state-value and advantage functions.
- **Application in Actor-Critic:** While these methods were originally proposed for DQN, similar ideas (e.g., using two critics) may help stabilize training in actor-critic algorithms.

### Bonus: Soft Actor-Critic (SAC) Implementation

- **Task:** Modify the existing codebase (starting from `rl.py`) to implement SAC.
- **Key Modifications:**
  1. **Actor Network Changes:**  
     The actor should output both a mean $$ \mu(s) $$ and a log standard deviation $$ \log \sigma(s) $$:
     $$
     \pi(s) \sim \mathcal{N}(\mu(s), \sigma(s)^2)
     $$
  2. **Entropy Regularization:**  
     The actor loss should incorporate an entropy term:
     $$
     L_{\text{actor}} = \mathbb{E}\left[\alpha \log \pi(a|s) - Q(s, a)\right]
     $$
  3. **Temperature Parameter:**  
     Introduce a learnable temperature parameter $$ \alpha $$ that controls the trade-off between reward maximization and exploration (entropy bonus). Optionally, use automatic tuning:
     $$
     L(\alpha) = -\alpha \, \mathbb{E}\left[\log \pi(a|s) + \mathcal{H}_{\text{target}}\right]
     $$
  4. **Critic Target Update:**  
     The target for the critic should also include the entropy term:
     $$
     \text{target } Q = r + \gamma \left( Q_{\text{target}}(s', a') - \alpha \log \pi(a'|s') \right)
     $$

- **Key Metrics:** Plot training and evaluation curves (episode reward vs. frames) to compare SAC against the baseline RL.

## Plots to Include

For **RL Training Only (Q1)**, include:
- **Training Curve:** $$ x: \text{Frame} $$ vs. $$ y: \text{Episode Reward} $$
- **Evaluation Curve:** $$ x: \text{Frame} $$ vs. $$ y: \text{Episode Reward} $$ (evaluation mode)

For **BC Training + RL Fine-Tuning (Q2)**, include:
- **Evaluation Curve Comparison:** Plot evaluation curves for RL-only vs. BC+RL.
- **Optional:** Plot offline BC loss vs. training iterations, if available.

For the **Bonus (SAC Implementation)**, include:
- **Training/Evaluation Reward Curves:** Similar to Q1.
- **Optional:** Plot the evolution of the temperature parameter $$ \alpha $$ vs. frames.

## How to Run

- Set up the environment using the provided `conda_env.yml`.
- Run the training scripts (e.g., `python train_rl.py ...` and `python train_bcrl.py ...`).
- Use the plotting scripts provided in this repository to generate the training and evaluation curves.
- Include the plots and your analysis in your PDF submission.

## Conclusion

This assignment explores off-policy RL and its enhancements through behavior cloning and SAC. The key takeaways include:
- **Sample Efficiency:** How behavior cloning pre-training can improve early performance.
- **Stability and Convergence:** How SAC’s entropy regularization aids exploration and stabilizes learning.
- **Technical Trade-offs:** Discussions on various RL variants and their potential improvements.

---

Feel free to modify this README to add any extra details or observations from your experiments. Enjoy coding and good luck with your submission!
