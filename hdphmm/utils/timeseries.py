#!/usr/bin/env python

import numpy as np
from statsmodels.tsa.api import VAR


class VectorAutoRegression:

    def __init__(self, timeseries, r):
        r""" Fit a vector autogressive (VAR) process to data using statsmodels.tsa.vector_ar. The output object is
        just reduction and renaming of attributes produced after running the fit() method of the VAR class

        For more detailed docs, see: https://www.statsmodels.org/dev/vector_ar.html#module-statsmodels.tsa.vector_ar

        For a multidimensional time series, one could write a system of dependent autoregressive equations:

        .. math::

            Y_t = A_1*Y_{t-1} + ... + A_p*Y_{t-p} + u_t

        where

        .. math::

           Y_t = \begin{bmatrix} y_{1,t} \\ y_{2,t} \\ ... \\  y_{k,t} \end{bmatrix},
           Y_{t-1} = \begin{bmatrix} y_{1,t-1} \\ y_{2,t-1} \\ ... \\  y_{k,t-1} \end{bmatrix},
           ...

        The matrices :math:`A_i` are K x K matrices where K is the number of dimensions of the trajectory.
        :math:`A_1` contains the 1st time lag autoregressive coefficients. If

        .. math::

            A_1 = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.4 \end{bmatrix}

        the associated system of equations for a VAR(1) process would be:

        .. math::

            y_{1,t} = 0.5y_{1,t-1} + u_{1,t}

            y_{2,t} = 0.4y_{2, t-1} + u_{2,t}

        Of course, adding cross-terms to A would create more complex dynamical behavior

        :math:`u_t` is a K-dimensional vector multivariate gaussian noise generated on the covariance matrix of the data

        :param timeseries: a T x K matrix where T is the number of observations and K is the number of variables/dimension
        :param r: autoregressive order. Number of past point on which current point depends

        :type timeseries: numpy.ndarray
        :type r: int
        """

        self.dim = timeseries.shape[1]  # number of dimensions
        self.order = r
        self.summary = None

        # fit VAR model with statsmodels.tsa
        model = VAR(timeseries)
        self.results = model.fit(r)

        # covariance matrix
        self.covariance = self.results.sigma_u_mle  # give same result as following commented out block
        # results summary stores the correlation matrix of residuals
        # https://blogs.sas.com/content/iml/2010/12/10/converting-between-correlation-and-covariance-matrices.html
        # corr = results.resid_corr  # residual correlation matrix
        # stds = results.resid.std(axis=0)  # standard deviation of data in each dimension
        # D = np.diag(stds)  # turn stds into a diagonal matrix
        # cov = D @ corr @ D  # convert correlation matrix to covariance matrix

        self.mu = self.results.params[0, :]

        self.mu_std = self.results.stderr[0, :]

        self.phi = self.results.coefs
        self.phi_std = np.zeros_like(self.phi)

        for i in range(self.dim):
            self.phi_std[:, i, :] = self.results.stderr[1:, i].reshape(r, self.dim)

    def summarize(self):

        self.summary = self.results.summary()


def switch_points(sequence):
    """ Determine points in discrete state time series where switches between states occurs. NOTE: includes first and
    last point of time series

    :param sequence: series of discrete states

    :type sequence: list or numpy.ndarray

    :return: list of indices where swithces between states occur
    :rtype: numpy.ndarray
    """

    # See https://stackoverflow.com/questions/36894822/how-do-i-identify-sequences-of-values-in-a-boolean-array
    switch_ndx = np.argwhere(np.diff(sequence)).squeeze().tolist()

    # add last frame as a switch point
    try:
        switch_ndx.append(len(sequence))
    except AttributeError:  # if there are no switches, it won't return a list
        switch_ndx = list([switch_ndx])
        switch_ndx.append(len(sequence))

    if switch_ndx[0] != 0:
        return np.array([0] + switch_ndx)  # also add first frame
    else:
        return np.array(switch_ndx)
