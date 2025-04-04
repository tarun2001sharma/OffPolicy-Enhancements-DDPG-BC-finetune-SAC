# import torch
# import torch.nn.functional as F

# import utils
# from agent.networks.actor import Actor
# from agent.networks.critic import Critic


# class Agent:
#     def __init__(
#         self,
#         obs_shape,
#         action_shape,
#         device,
#         lr,
#         hidden_dim,
#         critic_target_tau,
#         num_expl_steps,
#         update_every_steps,
#         stddev_schedule,
#         stddev_clip,
#         use_tb,
#     ):
#         self.device = device
#         self.lr = lr
#         self.critic_target_tau = critic_target_tau
#         self.update_every_steps = update_every_steps
#         self.use_tb = use_tb
#         self.num_expl_steps = num_expl_steps
#         self.stddev_schedule = stddev_schedule
#         self.stddev_clip = stddev_clip
#         self.bc_weight = 0.1 # applied to the behavior regularization term
        
#         # TODO: Define an actor network
#         self.actor = None


#         # TODO: Define a critic network and a target critic network
#         # Hint: The target critic network should be a copy of the critic network
#         self.critic = None
#         self.critic_target = None

#         # TODO: Define the optimizers for the actor and critic networks
#         self.actor_opt = None
#         self.critic_opt = None

#         self.train()
#         self.critic_target.train()

#     def __repr__(self):
#         return "bcrl"

#     def train(self, training=True):
#         self.training = training
#         self.actor.train(training)
#         self.critic.train(training)

#     def reinit_optimizers(self):
#         """
#         Reinitialize optimizers for RL after BC training
#         """
#         # TODO: Define the optimizers for the actor and critic networks
#         self.actor_opt = None
#         self.critic_opt = None

#         # For RL finetuning
#         self.critic_target.load_state_dict(self.critic.state_dict())

#     def act(self, obs, step, eval_mode):
#         obs = torch.as_tensor(obs, device=self.device).float()

#         stddev = utils.schedule(self.stddev_schedule, step)

#         # TODO: Get the action distribution from the actor network
#         dist = None

#         if eval_mode:
#             # TODO: Sample an action from the distribution in eval mode
#             action = None
#         else:
#             # If step is less than the number of exploration steps, sample a random action.
#             # Otherwise, sample an action from the distribution.
#             if step < self.num_expl_steps:
#                 # TODO: Sample a random action between -1 and 1
#                 pass
#             else:
#                 # TODO: Sample an action from the distribution
#                 action = None

#         return action.cpu().numpy()[0]

#     def update_bc(self, expert_replay_iter, step):
#         metrics = dict()

#         batch = next(expert_replay_iter)
#         obs, action = utils.to_torch(batch, self.device)
#         obs, action = obs.float(), action.float()

#         stddev = utils.schedule(self.stddev_schedule, step)

#         # TODO: Get the action distribution from the actor network
#         # and compute the actor loss
#         dist = None
#         actor_loss = None

#         # TODO: Optimize the actor network

#         if self.use_tb:
#             metrics["actor_loss"] = actor_loss.item()
            
#         return metrics

#     def update_critic(self, obs, action, reward, discount, next_obs, step):
#         metrics = dict()

#         with torch.no_grad():
#             stddev = utils.schedule(self.stddev_schedule, step)
#             # TODO: Compute the target Q value
#             # Hint: Use next obs and next action to compute the target Q value
#             target_Q = None

#         # TODO: Compute the Q value from the critic network
#         Q = None

#         # TODO: Compute the critic loss
#         critic_loss = None

#         # TODO: Optimize the critic network

#         if self.use_tb:
#             metrics["critic_target_q"] = target_Q.mean().item()
#             metrics["critic_loss"] = critic_loss.item()

#         return metrics

#     def update_actor(self, obs, obs_bc, action_bc, step):
#         metrics = dict()

#         stddev = utils.schedule(self.stddev_schedule, step)

#         # TODO: Get the action distribution from the actor network
#         # and sample an action from the distribution
#         dist = None
#         action = None

#         log_prob = dist.log_prob(action).sum(-1, keepdim=True)

#         # TODO: Get the Q value from the critic network
#         Q = None

#         # TODO: Compute the actor loss        
#         actor_loss = None

#         # behavior regularization for improved sample efficiency
#         # TODO: Compute a BC loss using observations and actions sampled from the expert
#         dist_bc = None
#         bc_loss = None
#         actor_loss += bc_loss * self.bc_weight

#         # TODO: Optimize the actor network

#         if self.use_tb:
#             metrics["actor_loss"] = actor_loss.item()
#             metrics["actor_logprob"] = log_prob.mean().item()
#             metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
#             # metrics["rl_loss"] = None # NOTE: Add if you want to log this value for debugging
#             metrics["bc_loss"] = bc_loss.item() * self.bc_weight

#         return metrics

#     def update_rl(self, replay_iter, expert_replay_iter, step):
#         metrics = dict()

#         if step % self.update_every_steps != 0:
#             return metrics

#         batch = next(replay_iter)
#         obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

#         # convert to float
#         obs = obs.float()
#         next_obs = next_obs.float()
#         action, reward, discount = action.float(), reward.float(), discount.float()
        
#         # For behavior regularization
#         batch = next(expert_replay_iter)
#         obs_bc, action_bc = utils.to_torch(batch, self.device)
#         obs_bc, action_bc = obs_bc.float(), action_bc.float()

#         if self.use_tb:
#             metrics["batch_reward"] = reward.mean().item()

#         # update critic
#         metrics.update(
#             self.update_critic(obs, action, reward, discount, next_obs, step)
#         )

#         # update actor
#         metrics.update(
#             self.update_actor(
#                 obs.detach(),
#                 obs_bc,
#                 action_bc,
#                 step,
#             )
#         )

#         # update critic target
#         utils.soft_update_params(
#             self.critic, self.critic_target, self.critic_target_tau
#         )

#         return metrics

#     def save_snapshot(self):
#         keys_to_save = ["actor", "critic"]
#         payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
#         return payload


import torch
import torch.nn.functional as F

import utils
from agent.networks.actor import Actor
from agent.networks.critic import Critic

class Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
    ):
        self.device = device
        self.lr = lr
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.bc_weight = 0.1  # applied to the behavior cloning loss
        
        # TODO: Define an actor network
        self.actor = Actor(obs_shape[0], action_shape, hidden_dim).to(device)

        # TODO: Define a critic network and a target critic network
        # Hint: The target critic network should be a copy of the critic network
        self.critic = Critic(obs_shape[0], action_shape, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape[0], action_shape, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # TODO: Define the optimizers for the actor and critic networks
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def __repr__(self):
        return "bcrl"

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def reinit_optimizers(self):
        """
        Reinitialize optimizers for RL finetuning after BC training.
        """
        # TODO: Define the optimizers for the actor and critic networks
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # For RL finetuning
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)

        # TODO: Get the action distribution from the actor network
        dist = self.actor(obs, stddev)
        if eval_mode:
            # TODO: Sample an action from the distribution in eval mode
            action = dist.mean
        else:
            # If step is less than the number of exploration steps, sample a random action.
            # Otherwise, sample an action from the distribution.
            if step < self.num_expl_steps:
                # TODO: Sample a random action between -1 and 1
                action = torch.rand(dist.mean.size(), device=self.device) * 2 - 1
            else:
                # TODO: Sample an action from the distribution
                action = dist.sample()
        return action.cpu().numpy()[0]

    def update_bc(self, expert_replay_iter, step):
        metrics = dict()
        batch = next(expert_replay_iter)
        obs, action = utils.to_torch(batch, self.device)
        obs = obs.float()
        action = action.float()

        stddev = utils.schedule(self.stddev_schedule, step)

        # TODO: Get the action distribution from the actor network
        # and compute the actor loss
        dist = self.actor(obs, stddev)
        # BC loss: mean squared error between the actor's predicted mean and the expert action.
        bc_loss = F.mse_loss(dist.mean, action)


        # TODO: Optimize the actor network
        self.actor_opt.zero_grad()
        bc_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            # Return both bc_loss and actor_loss as the same value, so logging in the training script works.
            metrics["actor_loss"] = bc_loss.item()
            metrics["bc_loss"] = bc_loss.item()
        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            # TODO: Compute the target Q value
            # Hint: Use next obs and next action to compute the target Q value
            stddev = utils.schedule(self.stddev_schedule, step)
            next_dist = self.actor(next_obs, stddev)
            next_action = next_dist.sample()
            target_Q = reward + discount * self.critic_target(next_obs, next_action)

        # TODO: Compute the Q value from the critic network
        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        # TODO: Optimize the critic network
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_loss"] = critic_loss.item()
        return metrics

    def update_actor(self, obs, obs_bc, action_bc, step):
    
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)

        # TODO: Get the action distribution from the actor network
        # and sample an action from the distribution
        # RL loss: maximize the Q value (minimize -Q)
        dist = self.actor(obs, stddev)
        action_rl = dist.sample()
        log_prob = dist.log_prob(action_rl).sum(-1, keepdim=True)

        # TODO: Get the Q value from the critic network
        Q = self.critic(obs, action_rl)
        rl_loss = -Q.mean()

        # behavior regularization for improved sample efficiency
        # TODO: Compute a BC loss using observations and actions sampled from the expert
        # BC loss: mimic the expert by matching the predicted mean to the expert action.
        dist_bc = self.actor(obs_bc, stddev)
        bc_loss = F.mse_loss(dist_bc.mean, action_bc)

        # TODO: Compute the actor loss        
        actor_loss = rl_loss + self.bc_weight * bc_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # TODO: Optimize the actor network
        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
            metrics["rl_loss"] = rl_loss.item()
            metrics["bc_loss"] = bc_loss.item() * self.bc_weight
        return metrics

    def update_rl(self, replay_iter, expert_replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        # convert to float
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        obs = obs.float()
        action = action.float()
        reward = reward.float()
        discount = discount.float()
        next_obs = next_obs.float()

        # For behavior regularization
        expert_batch = next(expert_replay_iter)
        obs_bc, action_bc = utils.to_torch(expert_batch, self.device)
        obs_bc = obs_bc.float()
        action_bc = action_bc.float()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic  and actor
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
        metrics.update(self.update_actor(obs.detach(), obs_bc, action_bc, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics

    def save_snapshot(self):
        keys_to_save = ["actor", "critic"]
        payload = {k: self.__dict__[k].state_dict() for k in keys_to_save}
        return payload
