#!/usr/bin/env python

import numpy as np
from statsmodels.tsa.api import VAR
import tqdm
from multiprocessing import Pool


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


def autocorrFFT(x):
    """ Function used for fast calculation of mean squared displacement

    :param x:
    :return:
    """

    N = len(x)
    F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real   # now we have the autocorrelation in convention B
    n = N*np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)

    return res / n  # this is the autocorrelation in convention A


def msd_fft(args):
    """ Calculate msd using a fast fourier transform algorithm

    :param x: trajectory of particle positions, equispaced in time
    :param axis: axis along which to calculate msd ({x:0, y:1, z:2})

    :type x: np.ndarray
    :type axis: int

    :return: msd as a function of time
    """

    x, axis = args

    r = np.copy(x)
    r = r[:, axis]

    if len(r.shape) == 1:
        r = r[:, np.newaxis]

    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
      Q = Q - D[m - 1] - D[N - m]
      S1[m] = Q / (N - m)

    return S1 - 2 * S2


def msd(x, axis, ensemble=False, nt=1):
    """ Calculate mean square displacement based on particle positions

    :param x: particle positions
    :param axis: axis along which you want MSD (0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2])
    :param ensemble: if True, calculate the ensemble MSD instead of the time-averaged MSD
    :param nt: number of threads to run on

    :type x: ndarray (n_frames, n_particles, 3)
    :type axis: int or list of ints
    :type ensemble: bool
    :type nt: int

    :return: MSD of each particle
    """

    frames = x.shape[0]  # number of trajectory frames
    ntraj = x.shape[1]  # number of trajectories
    MSD = np.zeros([frames, ntraj], dtype=float)  # a set of MSDs per particle

    size = len(x[0, :, axis].shape)  # number of axes in array where MSDs will be calculate

    if ensemble:

        for n in range(ntraj):  # start at 1 since all row 0 will be all zeros
            MSD[:, n] = ensemble_msd(x[0, n, axis], x[:, n, axis], size)

    else:
        if nt > 1:
            with Pool(nt) as pool:
                for i, t in enumerate(pool.map(msd_fft, [(x[:, n, :], axis) for n in range(ntraj)])):
                    MSD[:, i] = t
        else:
            for n in tqdm.tqdm(range(ntraj)):
                MSD[:, n] = msd_fft((x[:, n, :], axis))

    return MSD


def ensemble_msd(x0, x, size):

    if size == 1:

        return (x - x0) ** 2

    else:

        return np.linalg.norm(x0 - x, axis=1) ** 2


def bootstrap_msd(msds, N, confidence=68):
    """ Estimate error at each point in the MSD curve using bootstrapping

    :param msds: mean squared discplacements to sample
    :param N: number of bootstrap trials
    :param confidence: percentile for error calculation

    :type msds: np.ndarray
    :type N: int
    :type confidence: float
    """

    nT, nparticles = msds.shape

    msd_average = msds.mean(axis=1)

    eMSDs = np.zeros([nT, N], dtype=float)  # create n bootstrapped trajectories

    print('Bootstrapping MSD curves...')
    for b in tqdm.tqdm(range(N)):
        indices = np.random.randint(0, nparticles, nparticles)  # randomly choose particles with replacement
        for n in range(nparticles):
            eMSDs[:, b] += msds[:, indices[n]]  # add the MSDs of a randomly selected particle
        eMSDs[:, b] /= nparticles  # average the MSDs

    lower_confidence = (100 - confidence) / 2
    upper_confidence = 100 - lower_confidence

    limits = np.zeros([2, nT], dtype=float)  # upper and lower bounds at each point along MSD curve
    # determine error bound for each tau (out of n MSD's, use that for the error bars)
    for t in range(nT):
        limits[0, t] = np.abs(np.percentile(eMSDs[t, :], lower_confidence) - msd_average[t])
        limits[1, t] = np.abs(np.percentile(eMSDs[t, :], upper_confidence) - msd_average[t])

    return limits
