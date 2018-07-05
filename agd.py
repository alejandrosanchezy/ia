# Project: AGD

__author__ = "Alejandro Sánchez Yalí"
__copyright__ = "Copyright 2009, Planet Earth"
__credits__ = "Alejandro Sánchez Yalí"
__license__ = "GPL"
__maintainer__ = "Alejandro Sánchez Yalí"
__email__ = "asanchezyali@gmail.com"
__status__ = "Production"
__date__ = "lun 30 abr 2018 15:22:51 -05"

import numpy as np
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats

"""AGD

It is implemented to the descent gradient algorithm and is applied to a linear
regression problem.
"""


def gradient_descent(alpha: float, x: list, y: list, epsilon: float,
                     max_iter=1000) -> float:
    """
    This function allows calculating the parameter "theta" for the application
    of a data set of the form (x, y).
    Args:
        alpha: learning rate.
        x: input data.
        y: output data.
        epsilon: stop criterion.
        max_iter: Maximum number of iterations.

    Returns:
        gamma(float): Intercept of afin application.
        beta(float). Slope of afin applicaton.
    """

    converged = False
    iter = 0
    k = x.shape[0]

    gamma = np.random.random(x.shape[1])
    beta = np.random.random(x.shape[1])

    temp_error = sum([(y[i] - gamma - beta * x[i]) ** 2 for i in range(k)])

    while not converged:

        grad0 = 1 / k * sum([(y[i] - gamma - beta * x[i]) for i in range(k)])
        grad1 = 1 / k * sum(
            [(y[i] - gamma - beta * x[i]) * x[i] for i in range(k)])

        temp_gamma = gamma + alpha * grad0
        temp_beta = beta + alpha * grad1

        gamma = temp_gamma
        beta = temp_beta

        error_square = sum([(y[i] - gamma - beta * x[i])**2 for i in range(k)])

        if abs(temp_error - error_square) <= epsilon:
            print('The iteration converges: ', iter)
            converged = True

        temp_error = error_square
        iter += 1

        if iter == max_iter:
            print('Maximum iterations exceeded')
            converged = True

    return gamma, beta


if __name__ == '__main__':
    kwargs = dict(n_samples=200, n_features=1, n_informative=1,
                  random_state=0, noise=35)

    x, y = make_regression(**kwargs)
    print('Test set size: ', x.shape, y.shape)

    alpha = 0.01
    epsilon = 0.01

    gamma, beta = gradient_descent(alpha, x, y, epsilon, max_iter=1000)
    print('gamma= ', gamma, ', beta =', beta)

    slope, intercept, r, p, slope_std_error = stats.linregress(x[:, 0], y)
    print('Intercept =', intercept, ',slope =', slope)

    y_predict = 0
    for i in range(x.shape[0]):
        y_predict = gamma + beta * x

    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.savefig('agd.png')
    pylab.show()
    print('Done!')
