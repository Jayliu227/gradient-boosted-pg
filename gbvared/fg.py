import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.random.manual_seed(12345)


class Phi(nn.Module):
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
    def __init__(self):
        super(ZeroPhi, self).__init__()
        pass

    def forward(self, z):
        return (z * 0.0).sum(1)


class ControlVariate:
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
    return torch.sin(z ** 1.9) + 9.8 + z * 2.27


batch_size = 4000
indices = 20
mu = 2
std = 1

cv = ControlVariate()
mse = nn.MSELoss()
use_cv = 1.0

expectations = []
variances = []

# z ~ N(mu, std)
z = mu + torch.randn(batch_size, 1) * std

print('Start experiment: use control variate <{}>'.format('Yes' if use_cv > 0 else 'No'))

for i in range(indices):
    score = (z - mu) / (std ** 2)

    if use_cv > 0:
        # find functional gradient
        phi_value, phi_grad = cv.get_value(z)
        estimator = score * (f(z) - phi_value) + phi_grad
        fg = (-2 * (estimator * (phi_grad - score))).detach()

        # boosting
        phi = Phi(1, 48, 1)
        optim = torch.optim.Adam(phi.parameters(), lr=0.001, betas=(0.9, 0.999))
        for _ in range(5000):
            values = phi.forward(z)
            loss = mse(fg, values)
            optim.zero_grad()
            loss.backward()
            optim.step()

        cv.add_phi(phi, 0.005)

    # evaluate stats for new cv
    phi_value, phi_grad = cv.get_value(z)
    estimator = score * (f(z) - use_cv * phi_value) + use_cv * phi_grad

    expectation = estimator.mean()
    variance = (estimator ** 2).mean() - expectation ** 2

    print('Update {}: mean<{}> var<{}>'.format(i, expectation, variance))

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
