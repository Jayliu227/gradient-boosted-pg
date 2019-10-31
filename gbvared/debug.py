import random
import torch

import utils

# print(utils.compare_gradients('../our_grad', '../original_grad'))

plotter = utils.Plotter()
utils.load_data_to_plotter('reward_records_gb.txt', plotter)
plotter.plot()

