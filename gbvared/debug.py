import random
import torch
import torch.nn as nn

import utils
import control_variate as cv


# print(utils.compare_gradients('../our_grad', '../original_grad'))

# plotter = utils.Plotter()
# utils.load_data_to_plotter('reward_records_gb.txt', plotter)
# plotter.plot()

# for testing gradient boosting

class F(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(F, self).__init__()
        self.layer = nn.Linear(state_dim, action_dim)

    def forward(self, states):
        return self.layer(states)


class G(nn.Module):
    def __init__(self, state_dim):
        super(G, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, 1)
        )

    def forward(self, states):
        return self.layer(states) * 10.0


def L(phi, f, g, states, actions):
    phi_values, phi_grad_actions = phi.get_value(states, actions)
    f_values = f.forward(states)
    g_values = g.forward(states)

    estimator = f_values * (g_values - phi_values) + phi_grad_actions
    return (estimator ** 2).sum(1).mean()


def calc(phi, f, g, states, actions):
    phi_values, phi_grad_actions = phi.get_value(states, actions)
    f_values = f.forward(states)
    g_values = g.forward(states)

    A = f_values * (g_values - phi_values) + phi_grad_actions
    B = phi_grad_actions - f_values

    return 2 * (A * B).sum(1)


batch_dim = 500
state_dim = 6
action_dim = 4

f = F(state_dim, action_dim)
g = G(state_dim)
f.eval()
g.eval()

phi = cv.ControlVariate()
phi.add_base_func(cv.ZeroFunc(), 1.0)

mse = nn.MSELoss()

for _ in range(100):
    states = torch.randn(batch_dim, state_dim)
    actions = torch.randn(batch_dim, action_dim)

    # use boosting to reduce L
    fg = -calc(phi, f, g, states, actions).unsqueeze(-1).detach()

    new_phi = cv.BaseFunc(state_dim + action_dim, 32)
    optim = torch.optim.Adam(new_phi.parameters(), lr=0.03, betas=(0.9, 0.999))

    for _ in range(1000):
        new_phi_values = new_phi.forward(states, actions)
        loss = mse(new_phi_values, fg)
        optim.zero_grad()
        loss.backward()
        optim.step()

    phi.add_base_func(new_phi, 0.03)

    print(L(phi, f, g, states, actions))
