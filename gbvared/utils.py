import pandas as pd
import os
import torch

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