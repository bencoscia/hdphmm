# !/usr/bin/env python

import argparse
import numpy as np
import mdtraj as md
from scipy.stats import wishart, norm
from scipy import io
import tqdm
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import combinations, permutations
from LLC_Membranes.llclib import file_rw, rand, physical, topology  # TODO: get rid of LLC_Membranes dependency
from hdphmm import generate_timeseries as gent
import pickle

np.set_printoptions(precision=4, suppress=True)


def initialize():

    parser = argparse.ArgumentParser(description='Generate timeseries with different underlying models')

    parser.add_argument('-t', '--traj', type=str, help='Name of trajectory data structure to load.')
    parser.add_argument('-g', '--gro', type=str, help='Name of gro file associated with traj. Only necessary if using'
                                                      'a GROMACS trajectory')
    parser.add_argument('-res', '--residue', type=str, help='Name of residue whose center of mass will be tracked')

    parser.add_argument('-d', '--dim', nargs='+', type=int, help='dimensions of trajectory to use in analysis')
    parser.add_argument('-obs', '--obstype', default='AR', type=str, help="Model describing observation sequence")
    parser.add_argument('-r', '--order', default=1, type=int, help='Autoregressive order (number of time lags that '
                                                                   'yt depends on.')
    parser.add_argument('-p', '--prior', default='MNIW', type=str, help='Type of prior to use for generating DP '
                                                                        'parameters')
    parser.add_argument('-smax', '--max_states', default=20, type=int, help='The max number of states that can be '
                                                                            'found')
    parser.add_argument('-niter', '--niterations', default=2000, type=int, help='Number of iterations to perform '
                                                                                'sampling procedure')
    parser.add_argument('-load', '--load', action="store_true", help='Load saved arrays from previously run HDPHMM')
    parser.add_argument('-plot_traj', '--plot_traj', default=0, type=int, help='Number of trajectory to plot. I.e. if '
                        'there are 4 plots and you want the second plot, pass `-plot_traj 1`')

    ####### Trajectory Generation #######
    parser.add_argument('-nd', '--ndraws', default=2000, type=int, help='Number of time steps to take')
    parser.add_argument('-n', '--ntraj', default=4, type=int, help='Number of trajectories to generate')
    parser.add_argument('-f', '--format', default='mat', type=str, help='Format of data array (mat or npz)')
    # Define transition matrix. Either provide your own, or let this script generate one
    parser.add_argument('-T', '--transition_matrix', nargs='+', action='append', type=float, help='Define a transition '
                        'matrix. If provided, this will be used to determine the number of states. Each row of the '
                        'transition matrix should be passed with a separate -T flag')
    parser.add_argument('-s', '--nstates', default=3, type=int, help='Number of states to switch between')
    parser.add_argument('-slip', '--slip', default=0.01, type=float, help='Determines how frequently things will '
                        'switch states. A slip of 0 results in a transition matrix with only ones on the diagonal. '
                        'Higher values of slip will give smaller and smaller ratios of diagonals to the rest.')

    # Autoregressive parameters
    parser.add_argument('-phis', '--phis', nargs='+', action='append', type=float, help='Define autoregressive'
                        'coefficients for each state. Coefficients for each state should be passed in order with '
                        'separate -phis flags. If this is not specified, the coefficients will be randomly generated'
                        'for you.')
    # noise parameters
    parser.add_argument('-stds', '--stds', nargs='+', type=float, help='List of standard deviations of Gaussian white '
                                                                       'noise for each state.')
    # phantom state linking
    parser.add_argument('-l', '--link', action="store_true", help='Link together the independent trajectories with a'
                                                                  'phantom state in between.')
    parser.add_argument('-pl', '--phantom_length', default=100, type=int, help='Number of time steps for phantom state '
                                                                               'linkage')

    return parser


class PriorError(Exception):
    """ Raised if invalid prior specified """

    def __init__(self, message):
        super().__init__(message)


class ModelError(Exception):
    """ Raised if invalid model specified """

    def __init__(self, message):
        super().__init__(message)


class DfError(Exception):
    """ Raised if invalid number of degrees of freedom chosen"""

    def __init__(self, message):
        super().__init__(message)


class InfiniteHMM:

    def __init__(self, data, observation_model='AR', prior='MNIW', order=1, max_states=20, dim=None, save_com=True,
                 load_com=False, gro=None, res=None, difference=True, traj_no=None):
        """
        :param data: trajectories to analyze. If gro is not None, then this should be the name of a GROMACS trajectory \
         file (.xtc or .trr) TODO: describe what goes in this data structure
        :param observation_model: Model describing the observations (AR = autoregressive)
        :param prior: prior (MNIW : Matrix Normal inverse Wishart produces)
        :param order: for AR observation model, autoregressive order of data (order = 1 would be Y_t = phi*Y_{t-1} + c)
        :param max_states: maximum number of states
        :param dim: limit calculation to a specified dimension(s) given by integer indices or a list of integer indices
        :param gro: GROMACS coordinate file
        :param res: name of residue
        :param save_com: If True and you are loading a GROMACS trajectory, save the center of mass coordinates for \
        quicker loading in future usage

        :type data: str or object
        :type observation_model: str
        :type prior: str
        :type order: int
        :type max_states: int
        :type dim: None, int or list of int
        :type gro: str
        :type res: str
        :type save_com: True
        """

        com = None
        if load_com:
            com = load_object(data)
            print('Loaded center-of-mass coordinates')

        if type(dim) is int:  # this takes care of array shape issues
            dim = [dim]

        self.labels = None
        self.actual_T = None
        if isinstance(com, tuple) or gro is not None:

            if com is None:

                print("Loading GROMACS trajectory file %s..." % data, end='', flush=True)
                t = md.load(data, top=gro)
                print("Done!")

                residue = topology.Residue(res)  # get residue attributes

                ndx = [a.index for a in t.topology.atoms if a.residue.name == res]  # index of all residue atoms
                names = [a.name for a in t.topology.atoms if a.residue.name == res][
                        :residue.natoms]  # names of atoms in one residue
                mass = [residue.mass[x] for x in names]  # mass of atoms in order that they appear in file
                print('Calculating center of mass trajectories of residue %s' % residue.name)
                self.com = physical.center_of_mass(t.xyz[:, ndx, :], mass)  # determine center of mass trajectories

                self.dt = t.time[1] - t.time[0]

                if save_com:
                    file_rw.save_object((self.com, self.dt), 'com.pl')
                    print('Saved center-of-mass coordinates')

            else:
                self.com = com[0]
                self.dt = com[1]

            if difference:  # take first order difference
                self.trajectories = self.com[1:, ...] - self.com[:-1, ...]
                #self.trajectories = self.trajectories[:, np.newaxis, :]
                print('Took first order difference of center of mass trajectories')
            else:
                self.trajectories = self.com[..., dim]

        elif isinstance(data, gent.GenData):

            self.trajectories = data.traj
            self.labels = data.state_sequence

            self.actual_T = data.T

            self.dt = 1

        if traj_no is not None:

            if isinstance(traj_no, int):
                traj_no = [traj_no]

            self.trajectories = self.trajectories[:, traj_no, :]

        # determine data characteristics
        self.dimensions = self.trajectories.shape[2]
        self.nT = self.trajectories.shape[0]
        self.nsolute = self.trajectories.shape[1]

        print('Fitting %d %d dimensional trajectories assuming an autoregressive order of %d' %
              (self.nsolute, self.dimensions, order))

        self.observation_model = observation_model  # type of model (AR, Gaussian, SLDS)
        self.prior = prior  # prior for noise and autoregressive parameters
        self.order = order  # autoregressive order
        self.max_states = max_states  # truncate infinte states possiblites down to something finite

        K = np.linalg.inv(np.diag(0.1 * np.ones(self.dimensions * self.order)))  # approximate noise level of data
        self.meanSigma = np.eye(self.dimensions)
        self.Ks = 1  # truncation level for mode transition distribution
        self.m = self.dimensions * self.order

        self.prior_params = {}
        if self.prior == 'MNIW':  # Matrix-normal inverse-wishart prior
            self.prior_params['M'] = np.zeros([self.dimensions, self.m])
            self.prior_params['K'] = K[:self.m, :self.m]
        else:
            raise PriorError('The prior %s is not implemented' % self.prior)

        # stuff to do with prior
        self.prior_params['nu'] = self.dimensions + 2  # degrees of freedom.
        self.prior_params['nu_delta'] = (self.prior_params['nu'] - self.dimensions - 1) * self.meanSigma

        # sticky HDP-HMM hyperparameter settings
        self.a_alpha = 1
        self.b_alpha = 0.01
        self.a_gamma = 1  # global expected # of HMM states (affects \beta) -- TODO: play with this
        self.b_gamma = 0.01
        if self.Ks > 1:  # i think this only applies to SLDS
            self.a_sigma = 1
            self.b_sigma = 0.01
        self.c = 100
        self.d = 1
        self.type = 'HDP'  # hierarchical dirichlet process.
        self.resample_kappa = True  # use the sticky model

        # things that initializeStructs.m does #
        if self.observation_model == 'AR':

            dimu = self.prior_params['M'].shape[0]
            dimX = self.prior_params['M'].shape[1]

            # define data structure used to fit a vector autoregression to data (VAR)
            invSigma = np.zeros([dimu, dimu, self.max_states, self.Ks])  # inverse of covariance matrix of noise
            A = np.zeros([dimu, dimX, self.max_states, self.Ks])  # autoregressive coefficients
            mu = np.zeros([dimu, self.max_states, self.Ks])  # mean
            self.theta = dict(invSigma=invSigma, A=A, mu=mu)

            # Ustats - used for VAR fit
            card = np.zeros([self.max_states, self.Ks])
            xx = np.zeros([dimX, dimX, self.max_states, self.Ks])  # MATLAB collapses last dimensions if its one
            yx = np.zeros([dimu, dimX, self.max_states, self.Ks])
            yy = np.zeros([dimu, dimX, self.max_states, self.Ks])
            sumy = np.zeros([dimu, self.max_states, self.Ks])
            sumx = np.zeros([dimX, self.max_states, self.Ks])

            self.Ustats = dict(card=card, XX=xx, YX=yx, YY=yy, sumY=sumy, sumX=sumx)

            self.blockSize = np.ones([self.nsolute, self.nT - self.order], dtype=int)
            self.blockEnd = np.cumsum(self.blockSize, axis=1)

            # create matrix with lagged data
            self.X = np.zeros([self.nsolute, self.dimensions * self.order, self.trajectories.shape[0] - self.order])
            for i in range(self.nsolute):
                self.X[i, ...] = self.make_design_matrix(self.trajectories[:, i])

            self.trajectories = self.trajectories[order:, ...]

        else:
            raise ModelError('The observation model %s is not implemented' % self.observation_model)

        # Matrices used to keep track of transitions
        N = np.zeros([self.max_states + 1, self.max_states],
                     dtype=int)  # N(i,j) = number of z_t=i to z_{t+1}=j transitions. N(Kz+1,i)=1 for i=z_1.
        Ns = np.zeros([self.max_states, self.Ks])  # Ns(i,j) = number of s_t=j given z_t=i
        uniqueS = np.zeros([self.max_states, 1])
        M = np.zeros_like(N)
        barM = np.zeros_like(N)  # barM(i,j) = no. tables in restaurant i that considered dish j

        sum_w = np.zeros([1, self.max_states])
        self.stateCounts = dict(N=N, Ns=Ns, uniqueS=uniqueS, M=M, barM=barM, sum_w=sum_w)

        # hyperparameters
        self.hyperparams = {'alpha0_p_kappa0': 0.0, 'rho0': 0.0, 'gamma0': 0.0, 'sigma0': 0.0}

        # initialize transition matrix, initial distribution, emission weights and beta vector
        self.pi_z = np.zeros([self.max_states, self.max_states])  # transition matrix
        self.pi_s = np.zeros([self.max_states, self.Ks])  # emission weights
        self.pi_init = None  # initial distribution
        self.beta_vec = None
        self.s = np.zeros([self.nsolute, self.nT - self.order])
        self.z = np.zeros([self.nsolute, self.nT - self.order], dtype=int)  # will hold estimated states

        self.iteration = 0

    def make_design_matrix(self, observations):
        """ Create an (order*d , T) matrix of shifted observations. For each order create a trajectory shifted an
        additional time step to the right. Do this for each dimension.

        For example, given [[1, 2, 3, 4], [5, 6, 7, 8]] and an order of 2, we would expect an output matrix:

        [0 1 2 3]
        [0 5 6 7]
        [0 0 1 2]
        [0 0 5 6]

        :param observations: time series sequence of observations

        :type observations: np.ndarray (nobservation x dimension)

        :return X: design matrix
        """

        d = observations.shape[1]  # dimensions
        T = observations.shape[0]  # number of points in trajectory

        X = np.zeros([self.order * d, T])

        for lag in range(self.order):
            ii = d * lag
            indx = np.arange(ii, ii + d)
            X[indx, min(lag + 1, T):] = observations[:(T - (lag + 1)), :].T

        return X[:, self.order:]

    def _sample_hyperparams_init(self):
        """ Sample hyperparameters to start iterations. Reproduction of sample_hyperparams_init.m for AR case
        """

        self.hyperparams['alpha0_p_kappa0'] = self.a_alpha / self.b_alpha  # Gj concentration parameter
        self.hyperparams['gamma0'] = self.a_gamma / self.b_gamma  # G0 concentration parameter

        if self.stateCounts['Ns'].shape[1] > 1:  # this condition should not happen
            self.hyperparams['sigma0'] = self.a_sigma / self.b_sigma
        else:
            self.hyperparams['sigma0'] = 1

        if self.resample_kappa:
            self.hyperparams['rho0'] = self.c / (self.c + self.d)
        else:
            self.hyperparams['rho0'] = 0

    def _sample_hyperparams(self):
        """ Sample concentration parameters that define the distribution on transition distributions and mixture weights
        of the various model components.
        """

        alpha0_p_kappa0 = self.hyperparams['alpha0_p_kappa0']
        sigma0 = self.hyperparams['sigma0']

        N = self.stateCounts[
            'N']  # N(i, j) = no. z_t = i to z_{t+1} = j transitions in z_{1:T}. N(Kz+1, i) = 1 for i = z_1
        Ns = self.stateCounts[
            'Ns']  # Ns(i, k) = no. of observations assigned to mixture component k in mode i (i.e. # s_t = k given z_t =i)
        uniqueS = self.stateCounts['uniqueS']  # uniqueS(i) = sum_j Ns(i, j) = no of mixture components from HMM-state i
        M = self.stateCounts['M']  # M(i, j) = no. of tables in restaurant i serving dish k
        barM = self.stateCounts['barM']  # barM(i, j) = no. of tables in restaurant i considering dish k
        sum_w = self.stateCounts['sum_w']  # sum_w(i) = no. of overridden dish assignments in restaurant i

        Nkdot = N.sum(axis=1)
        Mkdot = M.sum(axis=1)
        Nskdot = Ns.sum(axis=1)
        barK = sum(barM.sum(axis=0) > 0)
        validindices = np.where(Nkdot > 0)[0]
        validindices2 = np.where(Nskdot > 0)[0]

        gamma0 = self.hyperparams['gamma0']

        if validindices.size == 0:
            alpha0_p_kappa0 = np.random.gamma(self.a_alpha) / self.b_alpha
            gamma0 = np.random.gamma(self.a_gamma) / self.b_gamma
        else:
            alpha0_p_kappa0 = gibbs_conparam(alpha0_p_kappa0, Nkdot[validindices], Mkdot[validindices], self.a_alpha,
                                             self.b_alpha, 50)
            gamma0 = gibbs_conparam(gamma0, barM.sum(), barK, self.a_gamma, self.b_gamma, 50)

        self.hyperparams['gamma0'] = gamma0

        # There is another loop here if Ks > 1 !!
        if Ns.shape[1] > 1:

            if validindices2.size == 0:
                sigma0 = np.random.gamma(self.a_sigma) / self.b_sigma
            else:
                sigma0 = gibbs_conparam(sigma0, Nskdot[validindices2], uniqueS[validindices2], self.a_sigma,
                                        self.b_sigma, 50)
        else:
            sigma0 = 1

        if self.resample_kappa:
            # resample self-transition proportion parameter:
            A = self.c + sum_w.sum()
            B = self.d + M.sum() - sum_w.sum()
            # rho0 = np.random.beta(A, B)
            rho0 = randdirichlet([A, B])[0]

        self.hyperparams['alpha0_p_kappa0'] = alpha0_p_kappa0
        self.hyperparams['sigma0'] = sigma0
        self.hyperparams['rho0'] = rho0

    def _sample_distributions(self):
        """ Sample the transition distributions pi_z, initial distribution pi_init, emission weights pi_s, and global
        transition distribution beta from the priors on these distributions.

        reproduction of sample_dist.m
        """

        # define alpha0 and kappa0 in terms of alpha0 + kappa0 and rho0
        alpha0 = self.hyperparams['alpha0_p_kappa0'] * (1 - self.hyperparams['rho0'])
        kappa0 = self.hyperparams['alpha0_p_kappa0'] * self.hyperparams['rho0']
        sigma0 = self.hyperparams['sigma0']
        gamma0 = self.hyperparams['gamma0']

        # in first iteration, barM is all zeros, so the output looks like pulls from beta distributions centered
        # at 1 / self.max_states

        # Draw beta_vec using Dirichlet process; G0 ~ DP(gamma, H)  H is a uniform base measure
        # self.beta_vec = np.random.dirichlet(self.stateCounts['barM'].sum(axis=0) + gamma0 / self.max_states)  # G0
        self.beta_vec = randdirichlet(self.stateCounts['barM'].sum(axis=0) + gamma0 / self.max_states)[:, 0]  # REMOVE

        N = self.stateCounts['N']
        Ns = self.stateCounts['Ns']
        for j in range(self.max_states):
            # sample rows of transition matrix based on G0, counts and sticky parameter
            # Gj ~ DP(alpha, G0)  -- this is the hierarchical part. If it were Gj ~ DP(gamma, H) this wouldn't work
            vec = alpha0 * self.beta_vec + N[j, :]
            vec[j] += kappa0  # here is the sticky part. This ends up weighting self-transitions pretty heavily
            # self.pi_z[j, :] = np.random.dirichlet(vec)
            # self.pi_s[j, :] = np.random.dirichlet(Ns[j, :] + sigma0 / self.Ks)
            self.pi_z[j, :] = randdirichlet(vec)[:, 0]  # REMOVE
            self.pi_s[j, :] = randdirichlet(Ns[j, :] + sigma0 / self.Ks)[:, 0]  # REMOVE

        # self.pi_init = np.random.dirichlet(alpha0 * self.beta_vec + N[self.max_states, :])
        self.pi_init = randdirichlet(alpha0 * self.beta_vec + N[self.max_states, :])[:, 0]  # REMOVE

    def _sample_theta(self):
        """ reproduction of sample_theta.m

        Sampling follows this paper:
        http://papers.nips.cc/paper/3546-nonparametric-bayesian-learning-of-switching-linear-dynamical-systems.pdf

        """

        nu = self.prior_params['nu']
        nu_delta = self.prior_params['nu_delta']
        store_card = self.Ustats['card']

        if self.prior == 'MNIW':

            invSigma = self.theta['invSigma']
            A = self.theta['A']

            store_XX = self.Ustats['XX']
            store_YX = self.Ustats['YX']
            store_YY = self.Ustats['YY']

            K = self.prior_params['K']
            M = self.prior_params['M']
            MK = M @ K  # @ symbol does matrix multiplication
            MKM = MK @ M.T

            for kz in range(self.max_states):
                for ks in range(self.Ks):

                    if store_card[kz, ks] > 0:

                        Sxx = store_XX[:, :, kz, ks] + K
                        Syx = store_YX[:, :, kz, ks] + MK
                        Syy = store_YY[:, :, kz, ks] + MKM
                        # https://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
                        SyxSxxInv = np.linalg.lstsq(Sxx.T, Syx.T, rcond=None)[0].T
                        Sygx = Syy - SyxSxxInv @ Syx.T
                        Sygx = (Sygx + Sygx.T) / 2

                    else:
                        Sxx = K
                        SyxSxxInv = M
                        Sygx = 0

                    # sample inverse wishart distribution to get covariance estimate
                    sqrtSigma, sqrtinvSigma = randiwishart(Sygx + nu_delta, nu + store_card[kz, ks])

                    invSigma[:, :, kz, ks] = sqrtinvSigma.T @ sqrtinvSigma  # I guess sqrtinvSigma is cholesky decomp

                    cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx)).T  # transposed to match MATLAB

                    # sample a matrix normal distribution to get AR parameter estimates
                    A[:, :, kz, ks] = sample_from_matrix_normal(SyxSxxInv, sqrtSigma, cholinvSxx)

            self.theta['invSigma'] = invSigma
            self.theta['A'] = A

    def inference(self, niter):
        """ Sample z and s sequences given data and transition distributions

        :param niter: number of iterations to run

        :type niter: int
        """

        self._sample_hyperparams_init()
        self._sample_distributions()
        self._sample_theta()

        for _ in tqdm.tqdm(range(niter)):
            self._update_ustats(self._sample_zs())
            self._sample_tables()
            self.iteration += 1
            self._sample_distributions()
            self._sample_theta()
            self._sample_hyperparams()

    def _sample_zs(self):
        """ reproduction of sample_zs.m

        :return:
        """

        N = np.zeros_like(self.stateCounts['N'])
        Ns = np.zeros_like(self.stateCounts['Ns'])

        obsIndzs = []
        for i in range(self.nsolute):
            obsIndzs.append(dict(tot=np.zeros([self.max_states, self.Ks], dtype=int),
                                 inds=np.zeros([self.max_states, self.Ks], dtype=object)))

        for i in range(self.nsolute):

            blockSize = self.blockSize[i, :]
            blockEnd = self.blockEnd[i, :]
            T = blockSize.size

            z = np.zeros([T], dtype=int)
            s = np.zeros([int(blockSize.sum())], dtype=int)

            likelihood = self._compute_likelihood(i)
            partial_marg = self._backwards_message_vec(likelihood)

            # sample the state and sub-state sequences
            totSeq = np.zeros([self.max_states, self.Ks], dtype=int)
            indSeq = np.zeros([T, self.max_states, self.Ks])

            for t in range(T):

                if t == 0:
                    Pz = np.multiply(self.pi_init.T, partial_marg[:, 0])
                    obsInd = np.arange(0, blockEnd[0])

                else:

                    Pz = np.multiply(self.pi_z[z[t - 1], :].T, partial_marg[:, t])
                    obsInd = np.arange(blockEnd[t - 1], blockEnd[t])

                Pz = np.cumsum(Pz)

                # beam sampling
                u = np.random.uniform()
                z[t] = (Pz[-1] * u > Pz).sum()  # removed addition of 1. States named from 0

                # add state to state counts matrix
                if t > 0:
                    N[z[t - 1], z[t]] += 1
                else:
                    N[self.max_states, z[t]] += 1  # store initial point in "root" restaurant

                for k in range(blockSize[t]):

                    if self.Ks > 1:
                        print('Ks > 1 untested in sample_zs!')
                        Ps = np.multiply(self.pi_s[z[t], :], likelihood[z[t], :, obsInd[k]])
                        Ps = np.cumsum(Ps)
                        s[obsInd[k]] = (Ps[-1] * np.random.uniform() > Ps).sum()  # removed addition of 1
                    else:
                        s[obsInd[k]] = 0

                    Ns[z[t], s[obsInd[k]]] += 1
                    totSeq[z[t], s[obsInd[k]]] += 1
                    indSeq[totSeq[z[t], s[obsInd[k]]] - 1, z[t], s[obsInd[k]]] = obsInd[k] + 1

            self.z[i, :] = z
            self.s[i, :] = s

            for j in range(self.max_states):
                for k in range(self.Ks):
                    obsIndzs[i]['tot'][j, k] = totSeq[j, k]
                    obsIndzs[i]['inds'][j, k] = sparse.csr_matrix(indSeq[:, j, k], dtype=int)

        binNs = np.zeros_like(Ns)
        binNs[Ns > 0] = 1
        self.stateCounts['N'] = N
        self.stateCounts['Ns'] = Ns
        self.stateCounts['uniqueS'] = binNs.sum(axis=1)

        return obsIndzs

    def _update_ustats(self, inds):
        """ reprduction of update_Ustats.m"""

        Ns = self.stateCounts['Ns']

        if self.observation_model == 'AR':

            unique_z = np.where(Ns.sum(axis=1) > 0)[0]  # indices of unique states that have been predicted

            dimu = self.trajectories.shape[2]
            dimX = self.X.shape[1]

            # reset these bois to zero
            self.Ustats['XX'] = np.zeros([dimX, dimX, self.max_states, self.Ks])
            self.Ustats['YX'] = np.zeros([dimu, dimX, self.max_states, self.Ks])
            self.Ustats['YY'] = np.zeros([dimu, dimu, self.max_states, self.Ks])
            self.Ustats['sumY'] = np.zeros([dimu, self.max_states, self.Ks])
            self.Ustats['sumX'] = np.zeros([dimX, self.max_states, self.Ks])

            for i in range(self.nsolute):

                u = self.trajectories[:, i, :].T
                X = self.X[i, ...]

                for kz in unique_z:
                    unique_s_for_z = np.where(Ns[kz, :] > 0)[0]
                    for ks in unique_s_for_z:
                        obsInd = inds[i]['inds'][kz, ks][:inds[i]['tot'][kz, ks]].data - 1  # yuck
                        # print(kz)
                        # print(obsInd)
                        # exit()
                        self.Ustats['XX'][:, :, kz, ks] += X[:, obsInd] @ X[:, obsInd].T
                        self.Ustats['YX'][:, :, kz, ks] += u[:, obsInd] @ X[:, obsInd].T
                        self.Ustats['YY'][:, :, kz, ks] += u[:, obsInd] @ u[:, obsInd].T
                        self.Ustats['sumY'][:, kz, ks] += u[:, obsInd].sum(axis=1)
                        self.Ustats['sumX'][:, kz, ks] += X[:, obsInd].sum(axis=1)

            self.Ustats['card'] = Ns

    def _sample_tables(self):
        """ reproduction of sample_tables.m
        """

        rho0 = self.hyperparams['rho0']
        alpha0 = self.hyperparams['alpha0_p_kappa0'] * (1 - rho0)
        kappa0 = self.hyperparams['alpha0_p_kappa0'] * rho0

        N = self.stateCounts['N']

        # sample M, where M(i, j) = number of tables in restaurant i served dish j
        alpha = self.beta_vec * np.ones([self.max_states, self.max_states]) * alpha0 + kappa0 * np.eye(self.max_states)
        alpha = np.vstack((alpha, alpha0 * self.beta_vec))
        M = randnumtable(alpha, N)

        barM, sum_w = sample_barM(M, self.beta_vec, rho0)

        self.stateCounts['M'] = M
        self.stateCounts['barM'] = barM
        self.stateCounts['sum_w'] = sum_w

    def _compute_likelihood(self, solute_no):
        """ compute the likelihood of each state at each point in the time series

        :param solute_no: solute number (trajectory number in self.trajectories)

        :type solute_no: int

        :return likelihood
        """

        if self.observation_model == 'AR':

            # Sample VAR parameters
            invSigma = self.theta['invSigma']
            A = self.theta['A']
            mu = self.theta['mu']

            dimu = self.trajectories.shape[2]
            T = self.trajectories.shape[0]

            log_likelihood = np.zeros([self.max_states, self.Ks, T])

            for kz in range(self.max_states):
                for ks in range(self.Ks):
                    cholinvSigma = np.linalg.cholesky(invSigma[:, :, kz, ks]).T
                    dcholinvSigma = np.diag(cholinvSigma)

                    # difference between trajectory and the data points predicted by current AR model
                    v = self.trajectories[:, solute_no, :].T - A[:, :, kz, ks] @ self.X[solute_no, ...] - \
                        mu[:, kz * np.ones([T], dtype=int), ks]

                    u = cholinvSigma @ v

                    log_likelihood[kz, ks, :] = -0.5 * np.square(u).sum(axis=0) + np.log(dcholinvSigma).sum()

            normalizer = log_likelihood.max(axis=0).max(axis=0)
            log_likelihood -= normalizer
            likelihood = np.exp(log_likelihood)  # can we just use log likelihoods?
            normalizer -= (dimu / 2) * np.log(2 * np.pi)

        return likelihood

    def _backwards_message_vec(self, likelihood):
        """ reproduction of backwards_message_vec.m

        :param likelihood: likelihood of being in each state at each time point (max_states x ks x T)

        :type likelihood: np.ndarray
        """

        T = self.trajectories.shape[0]
        bwds_msg = np.ones([self.max_states, T])
        partial_marg = np.zeros([self.max_states, T])

        block_like = likelihood.sum(axis=1)

        # compute messages backward in time
        for tt in range(T - 1, 0, -1):
            # multiply likelihood by incoming message
            partial_marg[:, tt] = np.multiply(block_like[:, tt], bwds_msg[:, tt])  # element-wise mult. of 2 20x1 arrays

            # integrate out z_t (??)
            bwds_msg[:, tt - 1] = self.pi_z @ partial_marg[:, tt]
            bwds_msg[:, tt - 1] /= bwds_msg[:, tt - 1].sum()

        # compute marginal for first time point
        partial_marg[:, 0] = np.multiply(block_like[:, 0], bwds_msg[:, 0])

        return partial_marg

    def summarize_results(self, cmap=plt.cm.jet, traj_no=0, plot_dim='all'):
        """ Plot estimated state sequence. If true labels exist, compare those.
        """

        # Get data
        estimated_states = self.z[traj_no, :]
        dim = self.trajectories.shape[2]  # number of dimensions of trajectory
        shift = 1.5 * self.trajectories[:, traj_no, :].max()

        found_states = list(np.unique(estimated_states))
        nT = len(estimated_states)

        state_counts = [list(estimated_states).count(s) for s in found_states]
        # print(state_counts)
        # print(estimated_states[:25])

        print('Found %d unique states' % len(found_states))

        if self.labels is not None:

            true_states = self.labels[self.order:, traj_no]
            true_state_labels = list(np.unique(true_states))
            nstates = len(true_state_labels)
            print('%d states were used to generate this data' % nstates)
            states = organize_states(true_states, true_state_labels, found_states, estimated_states)

        else:
            states = found_states

        # Print estimate properties
        estimated_transition_matrix = self.pi_z[tuple(np.meshgrid(states, states))].T

        # normalize so rows sum to 1
        for i in range(len(states)):
            estimated_transition_matrix[i, :] /= estimated_transition_matrix[i, :].sum()

        print('\nEstimated Transition Matrix:\n')
        print(estimated_transition_matrix)

        if self.labels is not None:

            actual_transition_matrix = self.actual_T
            rms = np.sqrt(np.square(estimated_transition_matrix - actual_transition_matrix).mean())

            print('\nActual Transition Matrix:\n')
            print(actual_transition_matrix)

            print('\nRoot mean squared error between estimated and true matrices: %.4f\n' % rms)

            # give extra states their own labels
            diff = len(found_states) - len(states)  # difference between no. of found states and actual no. of states
            extra_states = [x for x in found_states if x not in states]
            if diff > 0:
                for i in range(diff):
                    states.append(extra_states[i])
                    true_state_labels.append(nstates + i)

            M = dict(zip(true_state_labels, states))  # dictionary of state numbers mapped to original labels
            reverseM = dict(zip(states, true_state_labels))

            # determine the indices where the wrong label assignments occur
            wrong_label = []
            for s, estimate in enumerate(estimated_states):
                if estimate != M[true_states[s]]:
                    wrong_label.append(s)

            print('Correctly identified %.1f %% of states' % (100 * (1 - (len(wrong_label) / len(estimated_states)))))
        else:
            M = dict(zip(found_states, np.arange(len(found_states))))

        # MLE autoregressive coefficients and covariance matrices
        # print(self.theta['A'][..., states[2], 0])
        # print(np.linalg.inv(self.theta['invSigma'][..., states[2], 0]))

        # Make a color-coded plot

        # randomly assign color values from colormap
        colors = np.array([cmap(i) for i in np.random.choice(np.arange(cmap.N), size=len(found_states))])

        # for setting custom colors
        from matplotlib import colors as mcolors
        colors = np.array([mcolors.to_rgba(i) for i in
                           ['xkcd:blue', 'xkcd:orange', 'xkcd:red', 'xkcd:green', 'xkcd:yellow', 'xkcd:violet']])
        # colors = np.array([mcolors.to_rgba(i) for i in ['blue', 'blue', 'blue']])

        if self.labels is not None:

            fig, ax = plt.subplots(2, 1, figsize=(12, 8))

            # plot true state sequence
            z = true_states
            for i in range(dim):
                collection0 = multicolored_line_collection(np.arange(nT), self.trajectories[:, traj_no, i] + i * shift,
                                                           z, colors[:nstates, :])
                ax[0].add_collection(collection0)  # plot

            ax[0].set_title('True State Sequence', fontsize=16)
            ax[0].set_xlim([0, nT])
            ax[0].set_ylim([self.trajectories[:, traj_no, 0].min(), self.trajectories[:, traj_no, 1:].max() +
                            (dim - 1) * shift])  # min always based on 1st dimension since others are shifted up
            ax[0].tick_params(labelsize=14)

            z = np.array([reverseM[x] for x in estimated_states])

            ax_estimated = ax[1]

        else:

            fig, ax_estimated = plt.subplots(dim, 1, figsize=(12, 5), sharex=True)

            z = np.array([M[s] for s in estimated_states])

        # plot all found states with unique colors
        # plt.plot(self.com[:, traj_no, 2])
        # plt.show()
        # exit()
        # self.dt = 0.5
        #ax_estimated.set_title('Estimated State Sequence', fontsize=16)
        for i in range(dim):

            if self.labels is not None:

                ax_estimated.scatter(wrong_label, self.trajectories[wrong_label, traj_no, i] + i * shift, color='red',
                                     marker='x', zorder=10)

                ax_estimated.add_collection(multicolored_line_collection(np.arange(nT), self.trajectories[:, traj_no, i]
                                                                         + i * shift, z, colors))  # plot

            else:
                y = self.com[(1 + self.order):, traj_no, i]

                ax_estimated[i].add_collection(
                    multicolored_line_collection(np.arange(nT) * self.dt / 1000, y, z, colors))  # plot

                ax_estimated[i].set_xlim([0, nT * self.dt / 1000])
                ax_estimated[i].set_ylim([y.min(), y.max()])

        if self.labels is not None:
            ax_estimated.set_title('Estimated State Sequence', fontsize=16)
            ax_estimated.set_xlim([0, nT * self.dt])
            ymin = self.com[:, traj_no, 0].min()
            if dim > 1:
                ymax = self.com[:, traj_no, 1:].max() + (dim - 1) * shift
            else:
                ymax = self.com[:, traj_no, 0].max()

            ax_estimated.set_ylim([ymin, ymax])
            # ax_estimated.set_ylim([self.com[:, traj_no, 2].min(), self.com[:, traj_no, 2].max()])

            ax_estimated.tick_params(labelsize=14)
            ax_estimated.set_xlabel('Time', fontsize=14)
        else:
            ax_estimated[0].set_title('Estimated State Sequence', fontsize=16)
            ax_estimated[-1].set_xlabel('Time', fontsize=14)

        plt.tight_layout()
        plt.show()


def organize_states(true_states, true_state_labels, found_states, estimated_states):
    """ determine the optimal match between the estimated state sequence and the found state labels

    :param true_states:
    :param true_state_labels:
    :param found_states:
    :param estimated_states:
    :return:
    """

    # if too few states were found, add a dummy state in order to make following calculations work
    dummies = []
    if len(found_states) < len(true_state_labels):
        print('Less states were found than exist...adding a dummy state')
        for i in range(len(true_state_labels) - len(found_states)):
            dummy = sum(found_states)  # add all the found states to get a unique dummy state label
            found_states.append(dummy)
            dummies.append(dummy)

    # We need to identify which states match up with the known labels. I think the Munkres algorithm is a more
    # efficient way of doing this
    mismatches = []  # count up the number of mismatches for each subset and its permutations
    subsets = list(combinations(found_states, len(true_state_labels)))
    nperm = len(list(permutations(subsets[0])))
    for sub in subsets:
        p = permutations(sub)
        for i, perm in enumerate(p):
            M = dict(zip(true_state_labels, perm))
            wrong = 0
            for s, estimate in enumerate(estimated_states):
                if estimate != M[true_states[s]]:
                    wrong += 1
            mismatches.append(wrong)

    mindex = np.argmin(mismatches)  # index of minimum number of wrong label assignments
    subset = subsets[(mindex // nperm)]  # subset number of minimum wrong labels
    p = list(permutations(subset))  # permutations of subset
    states = list(p[mindex % nperm])  # the states listed in subset in the order that leads to minimum wrong labels

    return states


def multicolored_line_collection(x, y, z, colors):
    """ Color a 2D line based on which state it is in

    :param x: data x-axis values
    :param y: data y-axis values
    :param z: values that determine the color of each (x, y) pair
    """

    nstates = colors.shape[0]
    # come up with color map and normalization (i.e. boundaries of colors)

    cmap = ListedColormap(colors)
    bounds = np.arange(-1, nstates) + 0.1
    norm = BoundaryNorm(bounds, cmap.N)  # add

    # create line segments to color individually
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Set the values used for colormapping
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(2)

    return lc


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
    a = rand.randomwishart(df / 2, n)  # REMOVE

    sqrtinvx = (np.sqrt(2) * a) @ di
    sqrtx = np.linalg.inv(sqrtinvx).T

    return sqrtx, sqrtinvx


def sample_from_matrix_normal(M, sqrtV, sqrtinvK):
    """ reproduction of sampleFromMatrixNormal.m

    :param M:
    :param sqrtV:
    :param sqrtinvK:
    """

    mu = M.flatten(order='F')  # order F caused 1.5 days of debugging
    sqrtsigma = np.kron(sqrtinvK, sqrtV)

    # S = mu + sqrtsigma.T @ norm.rvs(size=mu.size)

    normald = rand.randomnormal(0, 1, mu.size)  # REMOVE
    S = mu + sqrtsigma.T @ normald  # REMOVE

    return S.reshape(M.shape, order='F')


def randnumtable(alpha, numdata):
    """ Reproduction of randnumtable.m

    :param alpha:
    :param numdata:
    :return:
    """

    numtable = np.zeros_like(numdata)
    for i in range(numdata.shape[1]):
        for j in range(numdata.shape[0]):
            if numdata[j, i] > 0:
                numtable[j, i] = 1 + sum(np.random.uniform(size=numdata[j, i] - 1) <
                                         (np.ones([numdata[j, i] - 1]) * alpha[j, i]) / (alpha[j, i] +
                                                                                         np.arange(1, numdata[j, i])))

    numtable[numdata == 0] = 0

    return numtable


def sample_barM(M, beta_vec, rho0):
    """ reproduction of sample_barM.m

    :param M: matrix of random table numbers
    :param beta_vec: G0 distribution pulled from a dirichlet process
    :param rho0: hyperparameter

    :type M: np.ndarray
    :type beta_vec: np.ndarray
    :type rho0: float

    :return barM
    :return sum_w
    """

    barM = np.copy(M)
    sum_w = np.zeros([M.shape[1]])

    for j in range(M.shape[1]):
        if rho0 > 0:
            p = rho0 / (beta_vec[j] * (1 - rho0) + rho0)
        else:
            p = 0

        # sum_w[j] = np.random.binomial(M[j, j], p)
        sum_w[j] = rand.randombinomial(M[j, j], p)  # REMOVE
        barM[j, j] = M[j, j] - sum_w[j]

    return barM, sum_w


def gibbs_conparam(alpha, numdata, numclass, aa, bb, numiter):
    """ Auxiliary variable resampling of DP concentration parameter. reproduction of gibbs_conparam.m

    :param alpha:
    :param numdata:
    :param numclass:
    :param aa:
    :param bb:
    :param numiter:
    """

    numgroup = numdata.size
    totalclass = numclass.sum()

    A = np.zeros([numgroup, 2])
    A[:, 0] = alpha + 1
    A[:, 1] = numdata

    for i in range(numiter):
        # beta auxiliary variables (the beta distribution is the 2D case of the dirichlet distribution)
        # xj = np.array([np.random.dirichlet(a) for a in A])
        xj = np.array([randdirichlet(a) for a in A])  # REMOVE

        xx = xj[:, 0]

        # binomial auxiliary variables -- debug this if there is an issue. I think this is right though
        zz = np.less(np.multiply(np.random.uniform(size=numgroup), alpha + numdata), numdata)

        gammaa = aa + totalclass - sum(zz)

        gammab = bb - sum(np.log(xx))

        # alpha = np.random.gamma(gammaa) / gammab
        alpha = (rand.randomgamma(gammaa) / gammab)[0, 0]  # REMOVE

    return alpha


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
        x = rand.randomgamma(a)
    except ValueError:
        a[a == 0] += 1e-16
        x = rand.randomgamma(a)

    x /= x.sum(axis=0)

    return x


def save_object(obj, filename):

    with open(filename, 'wb') as output:  # Overwrites any existing file.

        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):

    with open(filename, 'rb') as f:

        return pickle.load(f)


if __name__ == "__main__":

    args = initialize().parse_args()

    if not args.load:

        hmm = InfiniteHMM(args.traj, gro=args.gro, res=args.residue, observation_model=args.obstype, prior=args.prior,
                          order=args.order, max_states=args.max_states, dim=args.dim)
        hmm.inference(args.niterations)
        file_rw.save_object(hmm, 'hmm.pl')
        hmm.summarize_results(traj_no=args.plot_traj)  # make it pretty

    else:

        hmm = file_rw.load_object('hmm.pl')
        hmm.summarize_results(traj_no=args.plot_traj)

