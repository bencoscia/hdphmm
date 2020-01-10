#!/usr/bin/env python

#!/usr/bin/env python

"""
Generate random variates of various distributions based on draws from a uniform distribution
See: https://www.cse.wustl.edu/~jain/books/ftp/ch5f_slides.pdf

Given the same random seed, MATLAB and numpy will generate the same draws from a uniform distribution. This isn't the
case for other distributions. Instead, it's possible to generate other distributions based on U(0, 1) draws. In turn,
one can generate the same pseudorandom numbers in MATLAB and numpy using these functions and MATLAB equivalent functions
"""

import numpy as np
import math
np.set_printoptions(precision=4, suppress=True)


class DfError(Exception):
    """ Raised if invalid number of degrees of freedom chosen"""

    def __init__(self, message):
        super().__init__(message)


def randombeta(a, b, size=1):
    """ Generate random variates from beta distribution given shape parameters a, b
    """

    X = np.zeros([size])

    for i in range(size):
        X[i] = randombetavariate(a, b)

    return X


def randombetavariate(a, b):
    """ Generate single random variate from beta distribution given shape parameters a, b

    NOTE: only implemented for a < 0 and b < 0
    """

    if a < 1 and b < 1:
        x, y = [1, 1]  # make sure it enters while loop
        while x + y > 1:
            u1, u2 = np.random.uniform(size=2)
            x = u1 ** (1 / a)
            y = u2 ** (1 / b)

        return x / (x + y)

    # This doesn't work
    # elif isinstance(a, int) and isinstance(b, int):
    #
    #     u = sorted(np.random.uniform(size=(a + b + 1)))
    #
    #     return u[a - 1]


def randomexponential(a, size=1):
    """ Generate random variate from exponential distribution. Uses inverse CDF method

    pdf: f(x) = (1/a)e^(-x/a)

    :param a: scale paramter (a > 0)
    """

    u = np.random.uniform(size=size)

    return -a * np.log(u)


def randomgammaint(shape, scale=1.0, size=1):
    """ draw from random gamma distribution with integer shape parameter
    """

    U = np.random.uniform(size=(shape, size))
    X = -(1 / scale) * np.log(U).sum(axis=0)

    return X


def randomgamma(shape, scale=1.0, size=1, tol=1e-5):
    """ Generate random variates from a gamma distribution

    :param shape: Shape of gamma distribution (should be > 0)
    :param scale: Scale of the gamma distribution (should be > 0)
    :param size: number of variates to return

    :type shape: float or array_like of floats
    :type scale: float or array_like of floats
    :type size: int
    """

    if not isinstance(shape, (list, np.ndarray)):
        shape = [shape]

    rv = np.zeros([len(shape), size])

    for i, s in enumerate(shape):

        decimal, integer = math.modf(s)  # break up shape into integer and decimal

        if integer > tol:
            intgamma = randomgammaint(int(integer), scale=scale, size=size)
        else:
            intgamma = np.zeros([size])

        if decimal > tol:
            x = randombeta(decimal, 1 - decimal, size=size)
            y = randomexponential(1, size=size)
            decgamma = scale * x * y
        else:
            decgamma = np.zeros([size])

        rv[i, :] = intgamma + decgamma

    return rv


def randomnormal(mu, sigma, size=1):
    """ Generate random variate from the normal distribution using the Box Muller technique

    :return:
    """

    u = np.random.uniform(size=(2, size))

    x1 = mu + sigma*np.cos(2*np.pi*u[0, :]) * np.sqrt(-2*np.log(u[1, :]))
    #x2 = mu + sigma*np.sin(2*np.pi*u[0, :]) * np.sqrt(-2*np.log(u[1, :]))  # second independt Gaussian draw

    return x1


def randomwishart(a, d):

    sqrth = np.sqrt(0.5)
    norm = randomnormal(0, 1, d*d).reshape(d, d).T
    cholX = sqrth * np.triu(norm)
    i = np.arange(0, d)
    diag = [np.sqrt(randomgamma(g)) for g in a - i*0.5]
    for i in range(d):
        cholX[i, i] = diag[i]

    return cholX


def randombinomial(n, p):
    """ Generate random variates form a binomial(n, p) distribution

    :param n: number of trials
    :param p: probability of success
    """

    u = np.random.uniform(size=n)  # generate n U(0, 1) variates

    return sum(u < p)  # return number of variates less than p


def randomdirichlet(a):
    """ Python implementation of randdirichlet.m using randomgamma fucnction

    :param a: vector of weights (shape parameters to the gamma distribution)
    """

    x = randomgamma(a)
    x /= x.sum(axis=0)

    return x


def randiwishart(sigma, df):
    """ Generate an inverse Wishart random matrix in the form consistent with randiwishart.m

    :param sigma: covariance matrix (n x n)
    :param df: degrees of freedom. Must be greater than n (dimension of sigma)

    :type sigma: np.ndarray
    :type df: int

    :return: sqrtinvx
    :return: sqrtx
    """

    n = sigma.shape[0]
    if df < n:
        raise DfError('df < n. Please add degrees of freedom')

    d = np.linalg.cholesky(sigma)  # the output is the transpose of MATLAB's chol function
    di = np.linalg.inv(d)  # so no need to take transpose of d here

    # a = randwishart(df/2, n)
    a = randomwishart(df / 2, n)  # REMOVE

    sqrtinvx = (np.sqrt(2) * a) @ di
    sqrtx = np.linalg.inv(sqrtinvx).T

    return sqrtx, sqrtinvx


def randwishart(a, d):
    """ Implementation of randwishart.m in lightspeed toolbox. I think this is a wrong implementation but this is what
    is used

    :param a: degrees of freedom
    :param d: dimension of output matrix

    :type a: float
    :type d: int
    """

    sqrth = np.sqrt(0.5)
    cholX = sqrth * np.triu(np.random.normal(size=(d, d)))
    i = np.arange(0, d)
    diag = [np.sqrt(np.random.gamma(g)) for g in a - i * 0.5]
    for i in range(d):
        cholX[i, i] = diag[i]

    return cholX


def randdirichlet(a):
    """ Python implementation of randdirichlet.m using randomgamma fucnction

    :param a: vector of weights (shape parameters to the gamma distribution)
    """

    try:
        x = randomgamma(a)
    except ValueError:
        a[a == 0] += 1e-16
        x = randomgamma(a)

    x /= x.sum(axis=0)

    return x
