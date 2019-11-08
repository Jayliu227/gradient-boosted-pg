import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.random.manual_seed(12345)


class Phi(nn.Module):
    """Base phi function"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Phi, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, z):
        return self.layer(z)


class ZeroPhi(nn.Module):
    """Base phi function that always outputs 0"""
    def __init__(self):
        super(ZeroPhi, self).__init__()
        pass

    def forward(self, z):
        return (z * 0.0).sum(1)


class ControlVariate:
    """Phi wrapper class that can calculate value and gradient"""
    def __init__(self):
        self.phis = [ZeroPhi()]
        self.weights = [1.0]

    def add_phi(self, phi, weight):
        self.phis.append(phi)
        self.weights.append(weight)

    def get_value(self, z):
        values = torch.zeros(z.shape)
        grads = torch.zeros(z.shape)

        for (phi, weight) in zip(self.phis, self.weights):
            # we deal with one batch at a time
            for i, j in enumerate(z):
                j = j.unsqueeze(-1)
                j.requires_grad = True
                value = phi.forward(j)[0] * weight
                value.backward()
                values[i] += value
                grads[i] += j.grad.squeeze(0)

        return values, grads


def f(z):
    """Target function whose gradient of expectation we intend to evaluate"""
    return torch.sin(z) + 9.8 + z * 2.27


batch_size = 4000
# number of updates for boosting
indices = 20
# z ~ N(mu, std)
mu = 2
std = 1

cv = ControlVariate()
mse = nn.MSELoss()
# whether to use control variate or not
use_cv = 1.0
# FitQ method used in stein's paper
baseline = True

expectations = []
variances = []

# sample z
z = mu + torch.randn(batch_size, 1) * std

print('Start experiment: use control variate <{}>'.format('Yes' if use_cv > 0 else 'No'))

if baseline:
    """FitQ: surrogate loss is the mse between the target function and the phi function"""
    phi = Phi(1, 48, 1)
    optim = torch.optim.Adam(phi.parameters(), lr=0.001, betas=(0.9, 0.999))

    fz = f(z)
    loss = None

    for _ in range(5000):
        phi_value = phi.forward(z)
        loss = mse(phi_value, fz)
        optim.zero_grad()
        loss.backward()
        optim.step()

    score = (z - mu) / (std ** 2)
    wrap = ControlVariate()
    wrap.add_phi(phi, 1.0)
    phi_value, phi_grad = wrap.get_value(z)
    estimator = score * (fz - phi_value) + phi_grad

    expectation = estimator.mean()
    variance = (estimator ** 2).mean() - expectation ** 2

    print('Baseline FitQ: mean<{}> var<{}> phi_loss<{}>'.format(expectation, variance, loss))


for i in range(indices):
    """gradient boosting functional gradient descent"""
    score = (z - mu) / (std ** 2)

    loss = None
    if use_cv > 0:
        # find functional gradient
        phi_value, phi_grad = cv.get_value(z)
        estimator = score * (f(z) - phi_value) + phi_grad

        version = "Full"

        if version == "Full":
            # assume the hessian equals to the derivative
            fg = (-2 * (estimator * (phi_grad - score))).detach()
        elif version == "Half":
            # assume the hessian is always zero
            fg = (-2 * (estimator * - score)).detach()
        elif version == "None":
            # directly optimize the estimator (only for comparison)
            fg = estimator.detach()
        else:
            raise NotImplementedError

        # boosting
        phi = Phi(1, 48, 1)
        optim = torch.optim.Adam(phi.parameters(), lr=0.001, betas=(0.9, 0.999))
        for _ in range(5000):
            values = phi.forward(z)
            loss = mse(fg, values)
            optim.zero_grad()
            loss.backward()
            optim.step()

        cv.add_phi(phi, 0.002)

    # evaluate stats for new cv
    phi_value, phi_grad = cv.get_value(z)
    estimator = score * (f(z) - use_cv * phi_value) + use_cv * phi_grad

    expectation = estimator.mean()
    variance = (estimator ** 2).mean() - expectation ** 2

    print('Update {}: mean<{}> var<{}> phi_loss<{}>'.format(i, expectation, variance, loss))

    variances.append(variance)
    expectations.append(expectation)

# plot the variance and expectations
plt.subplot(2, 1, 1)
plt.plot(variances, '-+')
plt.title('Variance And Expectations With Different Phis')
plt.ylabel('var')

plt.subplot(2, 1, 2)
plt.plot(expectations, '-*')
if use_cv > 0:
    plt.xlabel('update times')
else:
    plt.xlabel('phi index')
plt.ylabel('mean')

plt.show()
