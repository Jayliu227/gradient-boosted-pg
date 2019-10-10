import random
import torch

import gbvared.control_variate as cv


c = cv.ControlVariate()

states = []
actions = []
for i in range(20):
    states.append(torch.rand(10))
    actions.append(torch.rand(5))

states = torch.stack(states)
actions = torch.stack(actions)

phi, action_grad = c.get_value(states, actions)

print(phi.shape)
print(action_grad.shape)

