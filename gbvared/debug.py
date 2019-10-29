import random
import torch

import utils

plotter = utils.Plotter()
utils.load_data_to_plotter('reward_records_1.txt', plotter)
plotter.plot()

