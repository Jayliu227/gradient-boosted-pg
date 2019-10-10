import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


class Plotter:
    def __init__(self):
        self.all_data = []

    def add_data(self, data):
        self.all_data.append(data)

    def plot(self, save=False, path='.', file_name='fig.png'):
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
