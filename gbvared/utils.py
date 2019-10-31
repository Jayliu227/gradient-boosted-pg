import pandas as pd
import os
import torch
import sys
from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self):
        self.all_data = []

    def add_data(self, data):
        self.all_data.append(data)

    def plot(self, save=False, path='.', file_name='fig.png'):
        """
        This will print the graph containing multiple runs of data
        :param save: whether to save the graph or not
        :param path: where to save the figure
        :param file_name: the name of the figure
        :return: None
        """
        min_len = min([len(data) for data in self.all_data])
        iterations = []
        rewards = []
        for data in self.all_data:
            iterations.extend([x for x in range(min_len)])
            rewards.extend(data[:min_len])
        data_dict = {'iterations': iterations, 'rewards': rewards}
        sns_plot = sns.lineplot(data=pd.DataFrame(data_dict), x='iterations', y='rewards')
        plt.show()

        if save:
            sns_plot.get_figure().savefig(os.path.join(path, file_name))


def calculate_function_grad(model, optimizer, phi, phi_grad_action, state, action, reward, cov_mat):
    """
        (1) f_score = ll_grad_w
        (2) f_advantage = A - phi
        (3) f_compensate = f_grad_w * phi_grad_action

        g_i = 2 * <f_score * f_advantage + f_compensate, f_score - f_compensate>
    """
    # (1)
    action_mean = model.actor(state)
    dist = MultivariateNormal(action_mean, cov_mat)
    ll = dist.log_prob(action)
    optimizer.zero_grad()
    ll.backward()
    f_score = flatten_grad(model.actor)

    # (2)
    f_advantage = model.critic(state) - reward - phi

    # (3)
    action_mean = model.actor(state)
    optimizer.zero_grad()
    (action_mean * phi_grad_action.detach()).sum().backward()
    f_compensate = flatten_grad(model.actor)

    return 2.0 * torch.dot(f_score * f_advantage + f_compensate, f_score - f_compensate)


def flatten_grad(model):
    """
    Given a model and return the flatten tensor of the gradients of the parameters
    """
    gradients = []
    for params in model.parameters():
        if params is None:
            print('Parameters do not have gradients.')
            sys.exit()
        gradients.append(params.grad.data.view(-1))
    return torch.cat(gradients)


def write_to_file_data(file_name, data):
    assert file_name is not None
    # create the file if not yet opened
    file = open(file_name, 'a+')
    file.write('%d\n' % data)
    file.close()


def load_data_to_plotter(file_name, plotter):
    assert file_name is not None
    file = open(file_name, 'r')
    if file.mode != 'r':
        print('Error: file could not be opened')
        return
    contents = file.read()

    # first print the data to the console
    print(contents)

    # add the data to the plotter the graph
    plotter.add_data([float(i) for i in contents.split('\n')[:-1]])

    file.close()


def gradient_norm(model):
    """
    :param model: the model for which we want to compute gradient norm
    :return: the l2 norm of the gradient
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm()
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def sample_action(action_means, action_std):
    return action_means + action_std * torch.randn(action_means.shape)


def save_gradients_to_files(gradient_name, model):
    assert gradient_name is not None
    gradients = []
    for params in model.parameters():
        if params.grad is None:
            print('Error: empty gradient.')
            return
        gradients.append(params.grad.data.view(-1))

    torch.save(torch.cat(gradients), '%s.pt' % gradient_name)


def load_gradients_from_file(gradient_name):
    assert gradient_name is not None
    return torch.load('%s.pt' % gradient_name)


def compare_gradients(grad_1, grad_2):
    grad_1 = load_gradients_from_file(grad_1)
    grad_2 = load_gradients_from_file(grad_2)
    return (grad_1 - grad_2).norm()