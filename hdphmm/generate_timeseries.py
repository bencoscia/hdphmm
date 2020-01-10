#!/usr/bin/env python

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse


def initialize():

    parser = argparse.ArgumentParser(description='Generate timeseries with different underlying models')

    parser.add_argument('-t', '--type', default='AR', type=str, help="Underlying dynamical model. Only AR is fully"
                                                                     "implemented.")
    parser.add_argument('-d', '--ndraws', default=2000, type=int, help='Number of time steps to take')
    parser.add_argument('-n', '--ntraj', default=4, type=int, help='Number of trajetories to generate')
    parser.add_argument('-f', '--format', nargs='+', default='npz', type=str, help='Format of output array (mat or npz)')
    parser.add_argument('-nd', '--ndimensions', default=1, type=int, help='Number of dimensions of trajectory.')

    # Define transition matrix. Either provide your own, or let this script generate one
    parser.add_argument('-T', '--transition_matrix', nargs='+', action='append', type=float, help='Define a transition '
                        'matrix. If provided, this will be used to determine the number of states. Each row of the '
                        'transition matrix should be passed with a separate -T flag')
    parser.add_argument('-s', '--nstates', default=3, type=int, help='Number of states to switch between')
    parser.add_argument('-slip', '--slip', default=0.01, type=float, help='Determines how frequently things will '
                        'switch states. A slip of 0 results in a transition matrix with only ones on the diagonal. '
                        'Higher values of slip will give smaller and smaller ratios of diagonals to the rest.')

    # Autoregressive parameters
    parser.add_argument('-r', '--order', default=1, type=int, help='Autoregressive order (number of time lags that '
                                                                   'yt depends on.')
    parser.add_argument('-phis', '--phis', nargs='+', action='append', type=float, help='Define autoregressive'
                        'coefficients for each state. Coefficients for each state should be passed in order with '
                        'separate -phis flags. If this is not specified, the coefficients will be randomly generated'
                        'for you.')

    # noise parameters
    parser.add_argument('-cov', '--covariance', nargs='+', action='append', type=float, help='Covariance matrix for '
                        'each state. Pass matrix for each state with a separate flag.')

    # phantom state linking
    parser.add_argument('-l', '--link', action="store_true", help='Link together the independent trajetories with a'
                                                                  'phantom state in between.')
    parser.add_argument('-pl', '--phantom_length', default=100, type=int, help='Number of time steps for phantom state '
                                                                               'linkage')

    return parser


class StateError(Exception):
    """ Raised if an undefined reaction is attempted """

    def __init__(self, message):

        super().__init__(message)


class GenData:

    def __init__(self, dynamics, dim, transition_matrix=None, phis=None, nstates=None, slip=0.25, order=1, cov=None,
                 stdmax=1):
        """ Given a transition matrix, generate timeseries using Markov Chains

        :param dynamics: underlying dynamical model ('Gaussian' or 'AR' are currently implemented)
        :param dim: number of dimensions of trajectory data
        :param transition_matrix: a list of N N-length lists. Each list represents a row of the transition matrix. If \
        None or False is passed, a transition matrix will be generated randomly
        :param phis: a list of N order-length lists of autoregressive coefficients for each state. In order of phi_1, \
        phi_2 etc. If None, these will be randomly generated
        :param nstates: number of states
        :param slip: determines ratio of diagonal elements to the rest. A higher 'high' will give you a smaller ratio \
        of diagonals to the rest and vice versa. A high of zero will result in an identity matrix, meaning there will \
        be no transitions from the initial state
        :param order: autoregressive order. Only specified if type is 'AR'
        :param cov: covariance matrix of multivariate Gaussian white noise for each state. If None, a random covariance
         matrix will be generated
        :param stdmax: maximum standard deviation of Gaussian white noise. This is only used if stds=None

        :type type: str
        :type dim: int
        :type transition_matrix: list of lists
        :type nstates: int
        :type slip: float
        :type order: int
        :type stds: list
        :type stdmax: float
        :type phis: list of lists
        """

        self.dynamics = dynamics

        self.T = None
        if transition_matrix is not None:
            self.T = np.array(transition_matrix)
            self.nstates = self.T.shape[0]  # define number of states based on provided transition matrix
        else:
            if not nstates:
                raise StateError("If no transition matrix is provided, the number of states must be specified")

            self.nstates = nstates
            self.generate_transition_matrix(slip)

        if dynamics == 'AR':

            self.phis = np.zeros([self.nstates, order, dim, dim])

            if phis is not None:
                # only works for r = 1
                for s in range(self.nstates):
                    try:
                        self.phis[s, ...] = np.array(phis[s]).reshape(dim, dim)
                    except IndexError:
                        raise IndexError('You have not provided enough phi matrices for the number of requested states')
            else:
                # NOTE: for multidimensional case, off-diagonal terms in each phi coefficient matrix are set to zero.
                # I'm not sure what the stabilty rules are for the multidimensional case
                self.phis = np.zeros([self.nstates, order, dim, dim])
                for s in range(self.nstates):
                    self.phis[s, ...] = generate_ar_parameters(order, dim)

        self.cov = np.zeros([self.nstates, dim, dim])
        if cov is None:
            for s in range(self.nstates):
                A = np.random.uniform(0, stdmax, size=(dim, dim))
                self.cov[s, ...] = A @ A.T
        else:
            for s in range(self.nstates):
                self.cov[s, ...] = np.array(cov[s]).reshape(dim, dim)

        self.mu = np.zeros([self.nstates, dim])

        self.state_labels = np.arange(self.nstates)

        self.traj = None
        self.state_sequence = None

    def generate_transition_matrix(self, high):
        """ generate a semi-random transition matrix

        :param high: determines ratio of diagonal elements to the rest. A higher 'high' will give you a smaller ratio of
        diagonals to the rest and vice versa. A high of zero will result in an identity matrix, meaning there will be no
        transitions from the initial state

        :type high: float
        """

        T = np.eye(self.nstates)  # start with identify matrix
        T += np.random.uniform(0, high, size=(self.nstates, self.nstates))  # add random draws from uniform distribution
        self.T = T / T.sum(axis=1, keepdims=1)  # normalize so sum of rows is 1

    def gen_trajectory(self, ndraws, ntraj):
        """ Generate time series with chosen underlying dynamics

        :param ndraws: number of sequential points to generate
        :return:
        """

        if self.dynamics == 'AR':

            return self._gen_ar_hmm(ndraws, ntraj)

    def _gen_ar_hmm(self, ndraws, ntraj):
        """ Generate a mean-zero autoregressive timeseries based on the transition matrix and autoregressive parameters.
        The timeseries is defined as:

        yt = \sum_{n=1}^{r} phi_n * y_{t-n} + \epsilon_t

        where r is autoregressive order and \epsilon_t is Gaussian white noise with state-dependent variance

        :param ndraws: number of points to generate for timeseries
        :param phis: autoregressive coefficients for each state (n_phis x n_states)

        :type ndraws: int
        :type phis: np.ndarray
        """

        order = self.phis.shape[1]
        dim = self.phis.shape[-1]
        data = np.zeros([ndraws + order, dim])
        state_labels = np.zeros([ndraws], dtype=int)
        self.state_sequence = np.zeros([ndraws, ntraj])
        self.traj = np.zeros([ndraws + order, ntraj, dim])

        for n in range(ntraj):

            state = np.random.choice(self.state_labels)  # choose initial state with uniform probability
            for d in range(order, ndraws):
                # choose state
                state = np.random.choice(self.state_labels, p=self.T[state, :])  # choose state based on transition matrix
                self.state_sequence[d, n] = state  # actual state labels for future comparison
                # state_labels[d] = state  # actual state labels for future comparison

                # calculate autoregressive terms
                #data[d, :] = sum([self.phis[state, i, ...] @ data[d - (i + 1), :] for i in range(order)])
                self.traj[d, n, :] = sum([self.phis[state, i, ...] @ data[d - (i + 1), :] for i in range(order)])

                # add Gaussian noise by drawing from multivariate normal distribution
                #data[d, :] += np.random.multivariate_normal(self.mu[state, ...], self.cov[state, ...])
                self.traj[d, n, :] += np.random.multivariate_normal(self.mu[state, ...], self.cov[state, ...])

        self.traj = self.traj[order:, ...]

        #return data[order:, :], state_labels


def generate_mar_process(phis, mu, cov, dim, ndraws):
    # TODO: move to generate_timeseries.py
    """ Create multivariate autoregressive timeseries. 'dim' dependent timeseries will be created with multivariate
    gaussian noise

    :param phis: autoregressive coefficients (order x dim x dim)
    :param mu: mean of trajectories in each dimension
    :param cov: covariance matrix describing multivariate normal distribution from which noise will be pulled
    :param dim: dimension, number of dependent trajectories to generate
    :param ndraws: number of points in each trajectory

    :type phis: numpy.ndarray
    :param mu: numpy.ndarray
    :type cov: numpy.ndarray
    :type dim: int
    :type ndraws: int
    """

    data = np.zeros([ndraws, dim])
    order = phis.shape[0]

    for d in range(order, ndraws):

        # calculate autoregressive terms
        data[d, :] = sum([phis[i, ...] @ data[d - (i + 1), :] for i in range(order)])

        # add gaussian noise
        data[d, :] += np.random.multivariate_normal(mu, cov)  # draw from multivariate normal distribution

    return data


def generate_ar_parameters(r, dim):
    """ generate autoregressive parameters, phi_n as defined below

    ** 1D case **

    yt = \sum_{n=1}^{r} phi_n * y_{t-n} + \epsilon_t

    NOTE: for stationarity to be achieved, the roots of 1 - phi_1*z - phi_2*z^2 ... phi_n * z^n must lie outside the
    unit circle

    ** multi-D case **

    For generating phi values for multi-dimensional trajectories, a matrix is used. Each matrix A_i contains the i-th
    lag autoregressive parameters for all dimension. i ranges from 1 to r. The matrices are square with shape dim x dim.
    Phis are generated for each dimension independently, so only the diagonals of A_i are filled.

    :param r: autoregressive order
    :param dim: number of dimensions

    :type r: int
    :type dim: int

    :return: vector of autoregressive coefficients, phi
    """

    phis = np.zeros([r, dim, dim])  # dim x dim matrix for each autoregressive order

    for i in range(dim):

        phi = None
        in_unit_circle = True
        while in_unit_circle:
            phi = np.random.uniform(-1, 1, size=r)  # made negative for compatibility with np.roots
            phi_transformed = [1] + (-1 * phi).tolist()
            roots = np.roots(phi_transformed[::-1])  # list reverse for compatibilty with np.roots
            in_unit_circle = False if np.absolute(roots).min() > 1 else True

        phis[:, i, i] = phi

    return phis


def link(t, labels, phantom_length=250, phantom_variance=0, phantom_mean=0):
    """ link together independent trajectories with a phantom state in between each

    :param t: trajectories (npoints, ntraj, 3)
    :param labels: true state of each point in trajectory (npoints, ntraj)
    :param phantom_length: number of timesteps to stay in phantom state
    :param phantom_variance: variance of phantom state
    :param phantom_mean: mean of phantom state

    :type t: np.ndarray
    :type labels: np.ndarray
    :type phantom_length: int
    :type phantom_variance: float
    :type phantom_mean: float

    :return single trajectory (npoints x ntraj + (ntraj - 1)*phantom_length, 3) and associated state labels
    """

    nstates = len(np.unique(state_labels.flatten()))

    phantom_state = phantom_variance*np.random.randn(phantom_length, 3) + phantom_mean
    p_label = nstates*np.ones(phantom_length)

    ntraj = t.shape[1]

    linked = t[:, 0, :]
    states = labels[:, 0]
    for i in range(1, ntraj):
        linked = np.concatenate((linked, phantom_state, t[:, i, :]), axis=0)
        states = np.concatenate((states, p_label, labels[:, i]), axis=0)

    return linked.reshape((linked.shape[0], 1, linked.shape[1])), states.reshape((states.shape[0], 1))


def save(array_dict, type, format):

    if 'mat' in format:
        io.savemat('%s_data.mat' % type.lower(), array_dict)
    if 'npz' in format:
        np.savez_compressed('%s_data.npz' % type.lower(), data=array_dict)


if __name__ == "__main__":

    args = initialize().parse_args()

    # generate trajectories
    data = np.zeros([args.ndraws, args.ntraj, args.ndimensions])
    state_labels = np.zeros([args.ndraws, args.ntraj], dtype=int)

    data_generator = GenData(args.type, args.ndimensions, args.transition_matrix, phis=args.phis, nstates=args.nstates,
                             slip=args.slip, order=args.order, cov=args.covariance)

    for i in range(args.ntraj):
        data[:, i, :], state_labels[:, i] = data_generator.gen_trajectory(args.ndraws)

    if args.link:
        linked_data, linked_state_labels = link(data, state_labels, phantom_length=args.phantom_length)
        save(dict(traj=data, labels=state_labels, T=data_generator.T, phis=data_generator.phis, linked_data=linked_data,
             linked_state_labels=linked_state_labels), args.type, args.format)
        # io.savemat('%s_data.mat' % args.type.lower(), dict(traj=data, labels=state_labels, T=data_generator.T,
        #                                                    phis=data_generator.phis, linked_data=linked_data,
        #                                                    linked_state_labels=linked_state_labels))

        plt.plot(linked_data[:, 0, 2])
        plt.plot(linked_state_labels)

        plt.show()

    else:
        # io.savemat('%s_data.mat' % args.type.lower(), dict(traj=data, labels=state_labels, T=data_generator.T,
        #                                                    phis=data_generator.phis))
        save(dict(traj=data, labels=state_labels, T=data_generator.T, phis=data_generator.phis), args.type, args.format)

        plt.plot(data[:, 0, 0])
        plt.plot(state_labels)

        plt.show()
