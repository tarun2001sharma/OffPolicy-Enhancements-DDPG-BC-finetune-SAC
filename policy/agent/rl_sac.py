
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

        self.actor = Actor(obs_shape[0], action_shape, hidden_dim).to(device)

        self.critic = Critic(obs_shape[0], action_shape, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape[0], action_shape, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Temperature (Alpha) inititialization
        # log_alpha is a learnable parameter; we exponentiate it to get alpha > 0
        self.log_alpha = torch.tensor([0.0], device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # target_entropy is set to -action_dim or some heuristic
        self.target_entropy = -action_shape[0]

        self.train()
        self.critic_target.train()

    @property
    def alpha(self):
        # Convert log_alpha to alpha
        return self.log_alpha.exp()

    def __repr__(self):
        return "rl_sac"

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        """
        Choose an action. In SAC, we do not rely on a fixed schedule
        for the actor's stddev — the actor learns log_std internally.
        """
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)

        # stddev = utils.schedule(self.stddev_schedule, step)  # not used

        # Build a Gaussian distribution from the actor (mean + log_std)
        dist = self.actor.get_dist(obs)  # see notes 

        if eval_mode:
            # For evaluation, take the mean (deterministic)
            action = dist.mean
        else:
            # If step < num_expl_steps, do random action in [-1, 1]
            if step < self.num_expl_steps:
                action = torch.rand(dist.mean.size(), device=self.device) * 2 - 1
            else:
                action = dist.sample()

        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        """
        Critic update with SAC target:
        target_Q =r + gamma *[Q_target(next_obs,   next_action)-alpha *log_prob]
        """
        metrics = {}

        with torch.no_grad():
            dist_next = self.actor.get_dist(next_obs)
            next_action = dist_next.sample()
            next_log_prob = dist_next.log_prob(next_action).sum(dim=-1, keepdim=True)

            # Q_target for next state (SAC includes entropy penalty)
            target_Q_val = self.critic_target(next_obs, next_action)
            target_Q = reward + discount * (target_Q_val - self.alpha * next_log_prob)

        current_Q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.use_tb:
            metrics["critic_loss"] = critic_loss.item()
            metrics["critic_target_q"] = target_Q.mean().item()

        return metrics

    def update_actor_and_alpha(self, obs, step):
        """
        Actor update in SAC:
        actor_loss = E[alpha *log_prob - Q]
        Also update alpha (temperature) to match a target entropy
        """
        metrics = {}

        dist = self.actor.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        Q_val = self.critic(obs, action)
        # Actor aims to minimize (alpha * log_prob - Q)
        # => maximize (Q - alpha * log_prob)
        actor_loss = (self.alpha * log_prob - Q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # temp Alpha Update
        # alpha_loss = E[-log_alpha *(log_prob + target_entropy).detach()]
        # doubt - maybe we only do this if we want automatic entropy tuning!!??
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["alpha_value"] = self.alpha.item()
            metrics["alpha_loss"] = alpha_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        obs = obs.float()
        next_obs = next_obs.float()
        action, reward, discount = action.float(), reward.float(), discount.float()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))
        metrics.update(self.update_actor_and_alpha(obs.detach(), step))
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def save_snapshot(self):
        keys_to_save = ["actor", "critic", "log_alpha"]
        payload = {}
        for k in keys_to_save:
            if hasattr(self.__dict__[k], "state_dict"):
                payload[k] = self.__dict__[k].state_dict()
            else:
                # For log_alpha (a tensor, not a module)
                payload[k] = self.__dict__[k]
        return payload

