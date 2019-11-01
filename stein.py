import torch
import torch.nn as nn
import gym
import numpy as np
from torch.distributions import MultivariateNormal

from gbvared import control_variate as cv
from gbvared import utils


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.means = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.means[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_std = action_std
        self.action_var = torch.full((action_dim,), action_std * action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # use policy to get our action means
        action_mean = self.actor(state)

        # construct covariance matrix
        cov_mat = torch.diag_embed(self.action_var)

        # construct the multivariate normal distribution used for finding logprob
        dist = MultivariateNormal(action_mean, cov_mat)

        # now the action generated by a noise and deterministic function
        action = utils.sample_action(action_mean, self.action_std)

        # get our action log probability
        action_logprob = dist.log_prob(action)

        # save the transitions
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.means.append(action_mean)

        return action.detach()

    def process(self, states, old_actions):
        # generate the current mean value vector
        action_means = self.actor(states)

        # construct the covariance matrices
        action_var = self.action_var.expand_as(torch.squeeze(action_means))
        cov_mat = torch.diag_embed(action_var)

        # construct the distribution out of the mean vectors and covariance matrices
        dist = MultivariateNormal(torch.squeeze(action_means), cov_mat)

        # get the log probability
        action_logprobs = dist.log_prob(torch.squeeze(old_actions))

        # get the entropy
        dist_entropy = dist.entropy()

        # get our V value
        state_values = self.critic(states)

        # make sure all these values are in batch format
        return action_means, action_logprobs, dist, torch.squeeze(state_values), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_var = torch.full((action_dim,), action_std * action_std)
        self.cov_mat = torch.diag_embed(self.action_var)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = ActorCritic(state_dim, action_dim, action_std)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.MseLoss = nn.MSELoss()

        self.beta = 1.0
        self.eta = 50
        self.kl_targ = 0.003

        # define control variate
        self.control_variate = cv.ControlVariate()
        # add our initial base function
        self.control_variate.add_base_func(func=cv.ZeroFunc(), weight=1.0)

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.policy.act(state, memory).data.numpy().flatten()

    def update(self, memory, update_time):
        assert self.K_epochs > 0

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).detach()
        old_means = torch.squeeze(torch.stack(memory.means)).detach()

        # construct old distributions
        old_dist = MultivariateNormal(old_means.detach(), self.cov_mat)

        # compute phi(s, a) and grad_phi w.r.t actions
        phi_value, phi_grad_action = self.control_variate.get_value(old_states, old_actions)

        use_cv = 1.

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            """
                Derivation of the update:
                Naming: X_grad_Y means the gradient of X w.r.t Y
                
                Assume that a = f(s) = mu + std * noise
                    f_grad_w = mu_grad_w + (std * noise)_grad_w = mu_grad_w
                    pi(a|s) = C * e^-(1/2(a - mu)^T*SIGMA^-1*(a - mu))
                    log_pi(a|s) = logC - (1/2) * (a - mu)^T*SIGMA^-1*(a - mu)
                    ll_grad_mu = SIGMA^-1 * (a - mu)
                    ratio = exp(logprobs - old_logprobs)
                
                loss = mu dot (ratio * (ll_grad_mu * (A - phi) + phi_grad_action)).detach()
                loss_grad_w = mu_grad_w dot ratio * (ll_grad_mu * (A - phi) + phi_grad_action) 
                            = ratio * (mu_grad_w dot ll_grad_mu * (A - phi) + mu_grad_w dot phi_grad_action)
                            = ratio * (mu_grad_w dot ll_grad_mu * (A - phi) + f_grad_w dot phi_grad_action)
                            = ratio * (ll_grad_w dot (A - phi) + f_grad_w dot phi_grad_action)
                            
                Dimensions:
                A               - [batch, 1]
                mu              - [batch, action_dim]
                phi             - [batch, 1]
                ratio           - [batch, 1]
                ll_grad_mu      - [batch, action_dim]
                phi_grad_action - [batch, action_dim]   
                
                We're instead using adaptive KL PPO.        
            """

            # Evaluating old actions and values
            action_means, logprobs, dist, state_values, dist_entropy = self.policy.process(old_states, old_actions)

            # calculate KL between the old distributions and new distributions
            kl = torch.distributions.kl_divergence(old_dist, dist).mean()

            # calculate the gradient of log likelihood of old actions w.r.t the action mean
            ll_grad_mu = (old_actions - action_means) / self.action_var

            # finding the ratio (pi_theta / pi_theta_old):
            ratios = torch.exp(logprobs - old_logprobs)

            # calculate advantages
            advantages = rewards - state_values

            # calculate surrogate loss
            surr = ratios.unsqueeze(-1) * (
                    ll_grad_mu * (advantages.unsqueeze(-1) - use_cv * phi_value) + use_cv * phi_grad_action)

            # dot product with the action mean to get our surrogate loss
            loss1 = -(action_means * surr.detach()).sum(1).mean()

            # critic loss
            loss2 = self.MseLoss(state_values, rewards)

            # kl loss
            loss3 = kl * self.beta + self.eta * max(0, kl - 2.0 * self.kl_targ)**2

            # total loss
            loss = loss1 + 0.5 * loss2 + loss3 - 0.01 * dist_entropy.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # early stopping if KL diverges badly
            if kl > self.kl_targ * 4:
                break

        # adaptive kl penalty:
        if kl > self.kl_targ * 2:
            self.beta = min(35., 1.5 * self.beta)
        elif kl < self.kl_targ / 2:
            self.beta = max(1 / 35, self.beta / 1.5)


def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v2"
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = 1234
    #############################################

    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    update_time = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory, update_time)
                memory.clear_memory()
                update_time += 1
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            utils.write_to_file_data('reward_records_stein.txt', running_reward)

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
