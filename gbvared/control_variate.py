import torch
import torch.nn as nn


class ControlVariate:
    def __init__(self):
        self.bases = []
        self.weights = []

    def get_value(self, states, actions):
        """
        :param states: states of the environment
                       [batch_size, state_dim]
        :param actions: actions from the policy
                       [batch_size, action_dim]
        :return:
               values: the value of the control variate
                       [batch_size, 1]
               action_grads: the gradient of the control variate w.r.t the actions
                       [batch_size, action_dim]
        """

        # init them to be zero
        values = torch.zeros((actions.shape[0], 1))
        action_grads = torch.zeros(actions.shape)

        for (base_func, weight) in zip(self.bases, self.weights):
            # we deal with one batch at a time
            for i, (state, action) in enumerate(zip(states, actions)):
                state = state.unsqueeze(0)
                action = action.unsqueeze(0)

                # make sure we can get the gradient of our actions
                action.requires_grad = True

                # calculate phi
                value = base_func.forward(state, action)[0] * weight
                # calculate gradient of phi with respect to actions
                value.backward()

                # update values and grads
                values[i] += value
                action_grads[i] += action.grad.squeeze(0)

        return values, action_grads

    def add_base_func(self, func, weight):
        self.bases.append(func)
        self.weights.append(weight)


class BaseFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BaseFunc, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = self.layers(x)
        return x


class ZeroFunc(nn.Module):
    def __init__(self):
        super(ZeroFunc, self).__init__()
        pass

    def forward(self, states, actions):
        # for now it is simply a zero function
        x = torch.cat([states, actions], dim=1) * 0.0
        x = torch.sum(x, dim=1).unsqueeze(-1)
        return x
