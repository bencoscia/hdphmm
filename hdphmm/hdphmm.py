# !/usr/bin/env python

import numpy as np
import mdtraj as md
import tqdm
from scipy import sparse
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from itertools import combinations, permutations
from hdphmm import generate_timeseries as gent
from hdphmm.utils import file_rw, physical_properties, random, timeseries
import pymbar

np.set_printoptions(precision=4, suppress=True)


class PriorError(Exception):
    """ Raised if invalid prior specified """

    def __init__(self, message):
        super().__init__(message)


class ModelError(Exception):
    """ Raised if invalid model specified """

    def __init__(self, message):
        super().__init__(message)


class PhantomState:
    """ An object that parameterizes the phantom state used to link independent trajectories
    """

    def __init__(self, d, r, length=100, cov=None, A=None, mu=None):

        self.params = dict()

        if cov is None:
            self.params['sigma'] = np.zeros([1, d, d, 1])
        else:
            self.params['sigma'] = cov

        if A is None:
            self.params['A'] = np.zeros([1, r, d, d, 1])
        else:
            self.params['A'] = A

        if mu is None:
            self.params['mu'] = np.zeros([1, d, 1])
        else:
            self.params['mu'] = mu

        self.params['T'] = np.ones([1, 1, 1])
        self.params['pi_init'] = np.ones([1])
        self.length = length

    def generate_trajectory(self):

        traj_generator = gent.GenARData(params=self.params)
        traj_generator.gen_trajectory(self.length, 1)

        return traj_generator.traj


class InfiniteHMM:

    def __init__(self, data, observation_model='AR', prior='MNIW-N', order=1, max_states=20, dim=None, save_com=True,
                 load_com=False, gro=None, res='MET', difference=True, traj_no=None, radial=False,
                 build_monomer='NAcarb11V', first_frame=0, link=False, parameterize_each=False, keep_xy=False,
                 com_savename='com.pl', state_sequence=None, seed_sequence=None, save_every=1, **kwargs):
        """
        :param data: trajectories to analyze. If gro is not None, then this should be the name of a GROMACS trajectory \
         file (.xtc or .trr) TODO: describe what goes in this data structure
        :param observation_model: Model describing the observations (AR = autoregressive)
        :param prior: prior (MNIW : Matrix Normal inverse Wishart produces)
        :param order: for AR observation model, autoregressive order of data (order = 1 would be Y_t = phi*Y_{t-1} + c)
        :param max_states: maximum number of states
        :param dim: limit calculation to a specified dimension(s) given by integer indices or a list of integer indices
        :param save_com: If True and you are loading a GROMACS trajectory, save the center of mass coordinates for \
        quicker loading in future usage
        :param load_com: load save center of mass trajectory. If True, you should be passing the pickled output from \
        using the save_com option to data.
        :param gro: GROMACS coordinate file
        :param res: name of residue
        :param difference: Take the first order difference of the timeseries
        :param traj_no: If there are multiple trajectories, specify which of the trajectories to use
        :param radial: replace the x and y coordinates of a MD trajectory with a radial coordinate with respect to
        the pore centers of an HII phase lyotropic liquid crystal membrane.
        :param build_monomer: name of monomer used to build membrane
        :param first_frame: First frame of trajectory to analyze
        :param link: link together multiple trajectories separated by a phantom state
        :param parameterize_each: perform a separate parameterization for each independent trajectory
        :param keep_xy: when parameterizing in terms of distance from pore center, keep radial coordinate in terms of \
        xy.
        :param com_savename: name by which to save center of mass trajectory if save_com is True
        :param save_every: save parameter estimates every x number of frames

        :type data: str or object
        :type observation_model: str
        :type prior: str
        :type order: int
        :type max_states: int
        :type dim: None, int or list of int
        :type save_com: bool
        :type load_com: bool
        :type gro: str
        :type res: str
        :type traj_no: list or int
        :type radial: bool
        :type build_monomer: str
        :type first_frame: int
        :type link: bool
        :type parameterize_each: bool
        :type keep_xy: bool
        :type com_savename: str
        :type save_every: int
        """

        self.observation_model = observation_model  # type of model (AR is the only option currently)
        self.prior = prior  # prior for noise and autoregressive parameters (MNIW is only option currently)
        self.order = order  # autoregressive order
        self.max_states = max_states  # truncate infinte states possiblites down to something finite
        self.iter = 0  # iteration counter
        self.save_every = save_every
        self.res = res

        self.fix_state_sequence = False
        if state_sequence is not None:
            self.fix_state_sequence = True

        self.seed_sequence = seed_sequence

        com = None
        if load_com:
            com = file_rw.load_object(data)
            print('Loaded center-of-mass coordinates')

        if isinstance(data, tuple):
            com = data

        if isinstance(dim, int):  # this takes care of array shape issues
            dim = [dim]

        self.labels = None
        self.actual_T = None
        if isinstance(com, tuple) or gro is not None:

            if com is None:

                # print("Loading GROMACS trajectory file %s..." % data, end='', flush=True)
                # t = md.load(data, top=gro)[first_frame:]
                # print("Done!")
                #
                # residue = physical_properties.Residue(res)  # get residue attributes
                #
                # ndx = [a.index for a in t.topology.atoms if a.residue.name == res]  # index of all residue atoms
                # names = [a.name for a in t.topology.atoms if a.residue.name == res][
                #         :residue.natoms]  # names of atoms in one residue
                # mass = [residue.mass[x] for x in names]  # mass of atoms in order that they appear in file
                # print('Calculating center of mass trajectories of residue %s' % residue.name)
                # self.com = physical_properties.center_of_mass(t.xyz[:, ndx, :], mass)  # determine center of mass trajectories
                t = None
                com = file_rw.load_object('trajectories/com_%s.pl' % res)
                self.com = com[0]

                if radial:

                    try:
                        sp = kwargs['spline_params']
                    except KeyError:
                        sp = {'npts_spline': 10, 'save': True, 'savename': 'spline.pl'}

                    #monomer = physical_properties.Residue(build_monomer)  # properties of monomer used to build system
                    monomer = None
                    radial_distances = self._get_radial_distances(t, monomer, sp, keep_xy=keep_xy)

                    if keep_xy:
                        self.com = np.concatenate((radial_distances, self.com[..., 2][..., np.newaxis]), 2)
                    else:
                        self.com = np.concatenate(
                            (radial_distances[..., np.newaxis], self.com[..., 2][..., np.newaxis]), 2)

                #self.dt = t.time[1] - t.time[0]
                self.dt = com[1]

                if save_com:
                    file_rw.save_object((self.com, self.dt), com_savename)
                    print('Saved center-of-mass coordinates')

            else:

                self.com = com[0]
                self.dt = com[1]  # time step in picoseconds

                if radial and keep_xy:
                    if self.com.shape[-1] != 3:
                        raise Exception('I am assuming you are working in 3 dimensions. You say you want to keep the xy'
                                        'coordinates but only gave a 2 dimensional center of mass array.')

            if difference:  # take first order difference
                self.trajectories = self.com[1:, ...] - self.com[:-1, ...]
                #self.trajectories = self.trajectories[:, np.newaxis, :]
                print('Took first order difference of center of mass trajectories')
            else:
                self.trajectories = self.com[..., dim]

        elif isinstance(data, object):#gent.GenARData):

            self.trajectories = data.traj
            # import scipy.io as io
            # io.savemat('test_traj.mat', dict(traj=self.trajectories))
            self.labels = data.state_sequence

            self.actual_T = data.T

            self.dt = 1

        if traj_no is not None:

            if isinstance(traj_no, int):
                traj_no = [traj_no]  # to maintain dimensions of self.trajectories

            self.trajectories = self.trajectories[:, traj_no, :]
            self.com = self.com[:, traj_no, :]

        if link:

            # currently only uses defaults. Can add functionality to allow new params
            phantom_state = PhantomState(self.trajectories.shape[2], self.order)

            linked_traj = self.trajectories[:, [0], :]
            linked_com = self.com[:, [0], :]
            for t in range(1, self.trajectories.shape[1]):
                linked_traj = np.concatenate((linked_traj, phantom_state.generate_trajectory()))
                linked_traj = np.concatenate((linked_traj, self.trajectories[:, [t], :]))
                linked_com = np.concatenate((linked_com, phantom_state.generate_trajectory()))
                linked_com = np.concatenate((linked_com, self.com[:, [t], :]))

            self.trajectories = linked_traj
            self.com = linked_com

        # determine data characteristics
        self.dimensions = self.trajectories.shape[2]
        self.nT = self.trajectories.shape[0]
        self.nsolute = self.trajectories.shape[1]

        print('Fitting %d %d dimensional trajectories assuming an autoregressive order of %d' %
              (self.nsolute, self.dimensions, order))

        K = np.linalg.inv(np.diag(0.1 * np.ones(self.dimensions * self.order)))  # approximate noise level of data
        self.meanSigma = np.eye(self.dimensions)
        self.Ks = 1  # truncation level for mode transition distribution
        self.m = self.dimensions * self.order

        self.prior_params = {}
        # MNIW-N: inverse Wishart on(A, Sigma) and normal on mu
        # MNIW: matrix normal inverse Wishart on (A, Sigma) with mean forced to 0
        if self.prior == 'MNIW':  # Matrix-normal inverse-wishart prior. Mean forced to zero

            self.prior_params['M'] = np.zeros([self.dimensions, self.m])
            self.prior_params['K'] = K[:self.m, :self.m]

        elif self.prior == 'MNIW-N':

            self.prior_params['M'] = np.zeros([self.dimensions, self.m])
            self.prior_params['K'] = K[:self.m, :self.m]

            if traj_no is None:

                traj_no = np.arange(self.nsolute)

            if len(traj_no) > 1:

                for i in range(len(traj_no)):
                    self.trajectories[:, i, :] -= self.trajectories.mean(axis=0)[i, :]  # np.zeros(self.dimensions)

                self.prior_params['mu0'] = np.zeros([self.dimensions])

                self.sig0 = np.zeros(self.dimensions)
                for i in range(self.dimensions):
                    self.sig0[i] = self.trajectories[..., i].flatten().std()

            else:

                minmax = np.array([self.trajectories[:, 0, :].min(axis=0), self.trajectories[:, 0, :].max(axis=0)])

                self.prior_params['mu0'] = minmax.mean(axis=0)
                self.sig0 = (minmax[1, :] - self.prior_params['mu0']) / 2

                # self.prior_params['mu0'] = self.trajectories[:, 0, :].mean(axis=0)
                # self.sig0 = (self.trajectories[:, 0, :] - self.prior_params['mu0']).std(axis=0)  # * 2

                print(self.sig0)
                print(self.prior_params['mu0'])

            self.prior_params['cholSigma0'] = np.linalg.cholesky(self.sig0 * np.eye(self.dimensions))
            self.prior_params['numIter'] = 50

        else:

            raise PriorError('The prior %s is not implemented' % self.prior)

        # stuff to do with prior
        self.prior_params['nu'] = self.dimensions + 2  # degrees of freedom.
        self.prior_params['nu_delta'] = (self.prior_params['nu'] - self.dimensions - 1) * self.meanSigma

        # sticky HDP-HMM hyperparameter settings
        self.a_alpha = 1
        self.b_alpha = 0.01
        self.a_gamma = 50  # global expected # of HMM states (affects \beta) -- TODO: play with this
        self.b_gamma = 0.1
        if self.Ks > 1:  # I think this only applies to SLDS
            self.a_sigma = 1
            self.b_sigma = 0.01
        self.c = 100
        self.d = 1
        self.type = 'HDP'  # hierarchical dirichlet process.
        self.resample_kappa = True  # use the sticky model

        if 'hyperparams' in kwargs:
            if kwargs['hyperparams'] is not None:
                self._override_hyperparameters(kwargs['hyperparams'])

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

        if self.fix_state_sequence:
            self.z = state_sequence

        if self.seed_sequence is not None:
            self.z = self.seed_sequence

        self.iteration = 0

        self.convergence = dict()
        self.convergence['A'] = []
        self.convergence['invSigma'] = []
        self.convergence['T'] = []
        self.convergence['pi_init'] = []
        self.convergence['mu'] = []
        self.convergence['kappa0'] = []
        self.convergence['gamma0'] = []
        self.convergence['nstates'] = []
        self.found_states = None
        self.clustered_state_sequence = None
        self.clustered_parameters = None

        self.converged_params = dict()

    def _override_hyperparameters(self, hyperparams):

        if 'mu0' in hyperparams:
            print('mu0 adjust from ', self.prior_params['mu0'], end='')
            self.prior_params['mu0'] = hyperparams['mu0']
            print(' to ', self.prior_params['mu0'])

        if 'a_gamma' in hyperparams:
            self.a_gamma = hyperparams['a_gamma']

        if self.prior == 'MNIW-N':
            if 'sig0' in hyperparams:
                self.sig0 = hyperparams['sig0']
                self.prior_params['cholSigma0'] = np.linalg.cholesky(self.sig0 * np.eye(self.dimensions))

            if 'scale_sig0' in hyperparams:
                print('Sig0 adjusted from ', self.sig0, end='')
                self.sig0 *= hyperparams['scale_sig0']
                self.prior_params['cholSigma0'] = np.linalg.cholesky(self.sig0 * np.eye(self.dimensions))
                print(' to ', self.sig0)

    def _get_radial_distances(self, t, monomer, spline_params, keep_xy=False):

        # pore_atoms = [a.index for a in t.topology.atoms if a.name in monomer.pore_defining_atoms and
        #               a.residue.name in monomer.residues]

        # spline = physical_properties.trace_pores(t.xyz[:, pore_atoms, :], t.unitcell_vectors,
        #               spline_params['npts_spline'], save=spline_params['save'], savename=spline_params['savename'])[0]
        spline = physical_properties.trace_pores(None, None,
                      spline_params['npts_spline'], save=spline_params['save'], savename=spline_params['savename'])[0]

        # nres = self.com.shape[1]
        # if keep_xy:
        #     radial_distances = np.zeros([t.n_frames, nres, 2])
        # else:
        #     radial_distances = np.zeros([t.n_frames, nres])
        # npores = spline.shape[1]
        # for f in tqdm.tqdm(range(t.n_frames), unit=' Frames'):
        #     d = np.zeros([npores, nres])
        #     for p in range(npores):
        #         print(physical_properties.radial_distance_spline(spline[f, p, ...], self.com[f, ...],
        #                                                    t.unitcell_vectors[f, ...], keep_xy=keep_xy))
        #         break
        #
        #         d[p, :] = physical_properties.radial_distance_spline(spline[f, p, ...], self.com[f, ...],
        #                                                              t.unitcell_vectors[f, ...], keep_xy=keep_xy)
        #
        #     radial_distances[f, :] = d[np.argmin(d, axis=0), np.arange(nres)]

        nframes = self.com.shape[0]
        nres = self.com.shape[1]
        npores = spline.shape[1]

        if keep_xy:
            radial_distances = np.zeros([nframes, npores, nres, 2])
        else:
            radial_distances = np.zeros([nframes, npores, nres])

        unitcell_vectors = file_rw.load_object('trajectories/unitcell_com_%s.pl' % self.res)
        ndx = np.zeros([nframes, nres])
        for f in tqdm.tqdm(range(nframes), unit=' Frames'):

            if keep_xy:
                d = np.zeros([npores, nres, 2])
            else:
                d = np.zeros([npores, nres])

            for p in range(npores):
                d[p, ...] = physical_properties.radial_distance_spline(spline[f, p, ...], self.com[f, ...],
                                                                       unitcell_vectors[f, ...], keep_xy=keep_xy)

            if keep_xy:

                r = np.zeros([npores, nres])
                for p in range(npores):
                    r[p, :] = np.linalg.norm(d[p, ...], axis=1)
                ndx[f, :] = np.argmin(r, axis=0)
                radial_distances[f, ...] = d
            else:

                ndx[f, :] = np.argmin(d, axis=0)
                radial_distances[f, :] = d

        modes = mode(ndx, axis=0)[0].astype(int)

        return radial_distances[:, modes, np.arange(nres), ...][:, 0, ...]

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
            alpha0_p_kappa0 = self.gibbs_conparam(alpha0_p_kappa0, Nkdot[validindices], Mkdot[validindices], self.a_alpha,
                                             self.b_alpha, 50)
            gamma0 = self.gibbs_conparam(gamma0, barM.sum(), barK, self.a_gamma, self.b_gamma, 50)

        self.hyperparams['gamma0'] = gamma0

        # There is another loop here if Ks > 1 !!
        if Ns.shape[1] > 1:

            if validindices2.size == 0:
                sigma0 = np.random.gamma(self.a_sigma) / self.b_sigma
            else:
                sigma0 = self.gibbs_conparam(sigma0, Nskdot[validindices2], uniqueS[validindices2], self.a_sigma,
                                        self.b_sigma, 50)
        else:
            sigma0 = 1

        if self.resample_kappa:
            # resample self-transition proportion parameter:
            A = self.c + sum_w.sum()
            B = self.d + M.sum() - sum_w.sum()
            # rho0 = np.random.beta(A, B)
            rho0 = random.randdirichlet([A, B])[0][0]
        else:
            rho0 = 0

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
        self.beta_vec = random.randdirichlet(self.stateCounts['barM'].sum(axis=0) + gamma0 / self.max_states)[:, 0]  # REMOVE

        N = self.stateCounts['N']
        Ns = self.stateCounts['Ns']

        for j in range(self.max_states):
            # sample rows of transition matrix based on G0, counts and sticky parameter
            # Gj ~ DP(alpha, G0)  -- this is the hierarchical part. If it were Gj ~ DP(gamma, H) this wouldn't work
            vec = alpha0 * self.beta_vec + N[j, :]
            vec[j] += kappa0  # here is the sticky part. This ends up weighting self-transitions pretty heavily
            # self.pi_z[j, :] = np.random.dirichlet(vec)
            # self.pi_s[j, :] = np.random.dirichlet(Ns[j, :] + sigma0 / self.Ks)
            self.pi_z[j, :] = random.randdirichlet(vec)[:, 0]  # REMOVE
            self.pi_s[j, :] = random.randdirichlet(Ns[j, :] + sigma0 / self.Ks)[:, 0]  # REMOVE

        # self.pi_init = np.random.dirichlet(alpha0 * self.beta_vec + N[self.max_states, :])
        self.pi_init = random.randdirichlet(alpha0 * self.beta_vec + N[self.max_states, :])[:, 0]  # REMOVE

        if self.iter % self.save_every == 0:

            self.convergence['T'].append(self.pi_z.copy())
            self.convergence['pi_init'].append(self.pi_init.copy())
            self.convergence['kappa0'].append(kappa0)
            self.convergence['gamma0'].append(gamma0)

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
                    sqrtSigma, sqrtinvSigma = random.randiwishart(Sygx + nu_delta, nu + store_card[kz, ks])

                    invSigma[:, :, kz, ks] = sqrtinvSigma.T @ sqrtinvSigma  # I guess sqrtinvSigma is cholesky decomp

                    cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx)).T  # transposed to match MATLAB

                    # sample a matrix normal distribution to get AR parameter estimates
                    A[:, :, kz, ks] = self.sample_from_matrix_normal(SyxSxxInv, sqrtSigma, cholinvSxx)

            #print(np.linalg.inv(invSigma[..., 0, 0]))
            #print(A[..., 0, 0])
            self.theta['invSigma'] = invSigma
            self.theta['A'] = A

            if self.iter % self.save_every == 0:
                self.convergence['A'].append(A.copy())
                self.convergence['invSigma'].append(invSigma.copy())

        elif self.prior == 'MNIW-N':

            invSigma = self.theta['invSigma']
            A = self.theta['A']
            mu = self.theta['mu']

            store_XX = self.Ustats['XX']
            store_YX = self.Ustats['YX']
            store_YY = self.Ustats['YY']
            store_sumY = self.Ustats['sumY']
            store_sumX = self.Ustats['sumX']

            K = self.prior_params['K']
            M = self.prior_params['M']
            MK = M @ K  # @ symbol does matrix multiplication
            MKM = MK @ M.T

            mu0 = self.prior_params['mu0']
            cholSigma0 = self.prior_params['cholSigma0']
            Lambda0 = np.linalg.inv(self.prior_params['cholSigma0'].T @ self.prior_params['cholSigma0'])
            theta0 = Lambda0 @ self.prior_params['mu0']

            dimu = nu_delta.shape[0]

            for kz in range(self.max_states):
                for ks in range(self.Ks):

                    if store_card[kz, ks] > 0:

                        for n in range(self.prior_params['numIter']):

                            Sxx = store_XX[:, :, kz, ks] + K
                            Syx = store_YX[:, :, kz, ks] + MK - mu[:, kz, ks][:, np.newaxis] @ store_sumX[:, kz, ks][np.newaxis, :]

                            Syy = store_YY[:, :, kz, ks] + MKM - \
                                  mu[:, kz, ks][:, np.newaxis] @ store_sumY[:, kz, ks][np.newaxis, :] - \
                                  store_sumY[:, kz, ks][:, np.newaxis] @ mu[:, kz, ks][np.newaxis, :] + \
                                  store_card[kz, ks] * mu[:, kz, ks][:, np.newaxis] @ mu[:, kz, ks][np.newaxis, :]

                            # https://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
                            SyxSxxInv = np.linalg.lstsq(Sxx.T, Syx.T, rcond=None)[0].T
                            Sygx = Syy - SyxSxxInv @ Syx.T
                            Sygx = (Sygx + Sygx.T) / 2

                            # sample inverse wishart distribution to get covariance estimate
                            sqrtSigma, sqrtinvSigma = random.randiwishart(Sygx + nu_delta, nu + store_card[kz, ks])

                            invSigma[:, :, kz, ks] = sqrtinvSigma.T @ sqrtinvSigma

                            cholinvSxx = np.linalg.cholesky(np.linalg.inv(Sxx)).T  # transposed to match MATLAB

                            # sample a matrix normal distribution to get AR parameter estimates
                            A[:, :, kz, ks] = self.sample_from_matrix_normal(SyxSxxInv, sqrtSigma, cholinvSxx)

                            Sigma_n = np.linalg.inv(Lambda0 + store_card[kz, ks]*invSigma[:, :, kz, ks])
                            a = store_sumY[:, kz, ks][:, np.newaxis] - A[:, :, kz, ks] @ store_sumX[:, kz, ks][:, np.newaxis]  # 2 x 1
                            b = invSigma[:, :, kz, ks] @ a
                            mu_n = Sigma_n @ (theta0[:, np.newaxis] + b)

                            mu[:, kz, ks] = mu_n[:, 0] + (np.linalg.cholesky(Sigma_n) @ random.randomnormal(0, 1, dimu))

                    else:

                        # sample inverse wishart distribution to get covariance estimate
                        sqrtSigma, sqrtinvSigma = random.randiwishart(nu_delta, nu)

                        invSigma[:, :, kz, ks] = sqrtinvSigma.T @ sqrtinvSigma  # I guess sqrtinvSigma is cholesky decomp

                        cholinvK = np.linalg.cholesky(np.linalg.inv(K)).T  # transposed to match MATLAB

                        # sample a matrix normal distribution to get AR parameter estimates
                        A[:, :, kz, ks] = self.sample_from_matrix_normal(M, sqrtSigma, cholinvK)

                        mu[:, kz, ks] = mu0 + cholSigma0.T @ random.randomnormal(0, 1, dimu)

            self.theta['invSigma'] = invSigma
            self.theta['A'] = A

            if self.iter % self.save_every == 0:
                self.convergence['A'].append(A.copy())
                self.convergence['invSigma'].append(invSigma.copy())
                self.convergence['mu'].append(mu.copy())

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
            self.convergence['nstates'].append(len(np.unique(self.z)))
            self.iter += 1

            #print(np.unique(self.z).shape)

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

                if self.fix_state_sequence:
                    z[t] = self.z[i, t]
                elif self.seed_sequence is not None and self.iter == 0:
                    z[t] = self.z[i, t]
                else:
                    z[t] = (Pz[-1] * u > Pz).sum()  # removed addition of 1. States named from 0

                # add state to state counts matrix
                if t > 0:
                    N[z[t - 1], z[t]] += 1
                else:
                    N[self.max_states, z[t]] += 1  # store initial point in "root" restaurant

                for k in range(blockSize[t]):

                    if self.Ks > 1:
                        print('Ks > 1 untested in _sample_zs!')
                        Ps = np.multiply(self.pi_s[z[t], :], likelihood[z[t], :, obsInd[k]])
                        Ps = np.cumsum(Ps)
                        s[obsInd[k]] = (Ps[-1] * np.random.uniform() > Ps).sum()  # removed addition of 1
                    else:
                        s[obsInd[k]] = 0

                    Ns[z[t], s[obsInd[k]]] += 1
                    totSeq[z[t], s[obsInd[k]]] += 1
                    indSeq[totSeq[z[t], s[obsInd[k]]] - 1, z[t], s[obsInd[k]]] = obsInd[k] + 1

            #if not self.fix_state_sequence:
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

        M = self.randnumtable(alpha, N)

        barM, sum_w = self.sample_barM(M, self.beta_vec, rho0)

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

    @staticmethod
    def sample_from_matrix_normal(M, sqrtV, sqrtinvK):
        """ reproduction of sampleFromMatrixNormal.m

        :param M:
        :param sqrtV:
        :param sqrtinvK:
        """

        mu = M.flatten(order='F')  # order F caused 1.5 days of debugging
        sqrtsigma = np.kron(sqrtinvK, sqrtV)

        # S = mu + sqrtsigma.T @ norm.rvs(size=mu.size)

        normald = random.randomnormal(0, 1, mu.size)  # REMOVE
        S = mu + sqrtsigma.T @ normald  # REMOVE

        return S.reshape(M.shape, order='F')

    @staticmethod
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
                                                                                             np.arange(1,
                                                                                                       numdata[j, i])))

        numtable[numdata == 0] = 0

        return numtable

    @staticmethod
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
            sum_w[j] = random.randombinomial(M[j, j], p)  # REMOVE
            barM[j, j] = M[j, j] - sum_w[j]

        return barM, sum_w

    @staticmethod
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
            xj = np.array([random.randdirichlet(a) for a in A])  # REMOVE

            xx = xj[:, 0]

            # binomial auxiliary variables -- debug this if there is an issue. I think this is right though
            zz = np.less(np.multiply(np.random.uniform(size=numgroup), alpha + numdata), numdata)

            gammaa = aa + totalclass - sum(zz)

            gammab = bb - sum(np.log(xx))

            #alpha = np.random.gamma(gammaa) / gammab
            alpha = (random.randomgamma(gammaa) / gammab)[0, 0]  # REMOVE

        return alpha

    def _get_params(self, traj_no=0, equil=None, recalculate_mle=False, quiet=True):
        """ Plot estimated state sequence. If true labels exist, compare those.
        """

        # Get data
        estimated_states = self.z[traj_no, :]  # estimated state sequence
        self.found_states = list(np.unique(estimated_states))  # all of the states that were identified

        if recalculate_mle:

            # get points at which state switches occur
            switch_points = timeseries.switch_points(estimated_states)

            sigma = np.zeros([1, self.dimensions, self.dimensions, switch_points.size])
            A = np.zeros([1, self.order, self.dimensions, self.dimensions, switch_points.size])

            # Loop through and calculate MLE VAR at for each segment
            for i in range(switch_points.size - 1):
                start = switch_points[i]
                end = switch_points[i + 1]
                print(end - start)
                var = timeseries.VectorAutoRegression(self.trajectories[start:end, traj_no, :], self.order)
                #if end - start > 100:

                print(var.covariance)
                print(var.phi, var.phi_std)
                #exit()

        else:

            found_states = list(np.unique(estimated_states))  # all of the states that were identified

            # Print estimate properties
            block = tuple(np.meshgrid(self.found_states, self.found_states))
            estimated_transition_matrix = self.pi_z[block].T

            # normalize so rows sum to 1
            for i in range(len(found_states)):
                estimated_transition_matrix[i, :] /= estimated_transition_matrix[i, :].sum()

            if not quiet:

                print('Found %d unique states' % len(self.found_states))

                print('\nEstimated Transition Matrix:\n')
                print(estimated_transition_matrix)

            # s = int(round(estimated_states[275:1000].mean()))
            # print(self.convergence['A'][-1][..., s, 0])

            sigma = np.array(self.convergence['invSigma'])[..., self.found_states, 0]
            # sigma = sigma[::self.save_every, ...]

            # reorganize autoregressive parameter
            A = np.zeros([sigma.shape[0], self.order, self.dimensions, self.dimensions, len(self.found_states)])

            for r in range(self.order):
                A[:, r, ...] = np.array(self.convergence['A'])[:, :, r*self.dimensions:(r+1)*self.dimensions, self.found_states, 0]

            T = np.zeros([A.shape[0], len(self.found_states), len(self.found_states)])
            for i in range(T.shape[0]):
                T[i, ...] = self.convergence['T'][i][block].T
                for j in range(len(self.found_states)):
                    T[i, j, :] /= T[i, j, :].sum()
                    sigma[i, ..., j] = np.linalg.inv(sigma[i, ..., j])

            pi_init = np.array(self.convergence['pi_init'])[:, self.found_states]

            if equil is None:  # an attempt to automatically detect when the AR parameters are equilibrated

                equil = 0

                for s in range(len(self.found_states)):
                    for u in range(self.dimensions):
                        for x in range(self.dimensions):
                            equils = []
                            for r in range(self.order):
                                equils.append(pymbar.timeseries.detectEquilibration(A[:, r, u, x, s])[0])
                            equils.append(pymbar.timeseries.detectEquilibration(sigma[:, u, x, s])[0])
                            if max(equils) > equil:
                                equil = max(equils)

            self.converged_params = dict(A=A[equil:, ...], sigma=sigma[equil:, ...], T=T[equil:, ...],
                                         pi_init=pi_init[equil:, :])

            if self.prior == 'MNIW-N':
                self.converged_params['mu'] = np.array(self.convergence['mu'])[equil:, :, self.found_states, 0]

            # Detect equilibration on transition matrix and initial state distribution vector
            # for i in range(len(found_states)):
            #     for j in range(len(found_states)):
            #             equilT = pymbar.timeseries.detectEquilibration(T[:, i, j])[0]
            #             if equilT > equil:
            #                 equil = equilT
            #     equil_pi = pymbar.timeseries.detectEquilibration(pi_init[:, i])[0]
            #     if equil_pi > equil:
            #         equil = equil_pi

            if not quiet:
                print('\nAutoregressive parameters equilibrated after %d iterations' % equil)

    def subtract_mean(self, traj_no=0, simple_mean=False, return_dwells_hops=False):
        """ Calculate MLE params of each segment independently and subtract mean

        Currently only works for AR(1)

        Try AR and just mean.

        """

        estimated_states = self.z[traj_no, :]  # estimated state sequence
        dwells = []
        hops = []

        # get points at which state switches occur
        switch_points = timeseries.switch_points(estimated_states)

        # sigma = np.zeros([1, self.dimensions, self.dimensions, switch_points.size])
        # A = np.zeros([1, self.order, self.dimensions, self.dimensions, switch_points.size])

        zeroed = np.zeros_like(self.trajectories[:, traj_no, :])

        # Loop through and calculate MLE VAR at for each segment
        for i in range(switch_points.size - 1):
            start = switch_points[i]
            end = switch_points[i + 1]
            data = self.trajectories[start:end, traj_no, :]

            if (end - start) <= 4 or simple_mean:  # I think 4 can be replaced by (self.order + 1) * data.shape[1]
                # simple mean of data
                if i > 0:
                    hops.append(data.mean(axis=0) - mu)
                dwells.append(end - start)
                mu = data.mean(axis=0)

            else:

                # unconditional mean of MLE VAR fit to data sequence
                var = timeseries.VectorAutoRegression(self.trajectories[start:end, traj_no, :], self.order)
                mu = np.linalg.inv(np.eye(self.dimensions) - var.phi[0, ...]) @ var.mu

            zeroed[start:end] = data - mu

        if return_dwells_hops:
            return zeroed, dwells, hops
        else:
            return zeroed

    def reassign_state_sequence(self, clusters, labels=None):
        """ Reassign the state sequence, update transition matrix and finalize parameters after parameter clustering

        NOTE that applying this to more than one trajectory is implemented but not tested.

        :param clusters: Cluster object from cluster.py
        :param labels: if the Cluster object was created from multiple InfiniteHMM objects, specify the labels which \
        belong to the state sequence being reassigned. If None, it assumes len(clusters.labels) is equal to the number\
        of found states.

        :type clusters: class
        :type labels: NoneType or list
        """

        # map cluster numbers to state numbers.
        # this mapping assumes that the clusters are ordered in the same way as the identified states.

        if labels is None:
            labels = clusters.labels

        map_states = dict()
        for i, label in enumerate(labels):
            map_states[self.found_states[i]] = label

        # reassign state sequence
        self.clustered_state_sequence = np.zeros_like(self.z)
        ntraj, nT = self.z.shape
        for t in range(ntraj):
            for i in range(nT):
                self.clustered_state_sequence[t, i] = map_states[self.z[t, i]]

        # unique_labels = np.unique(clusters.labels)  # we need to include all labels in clusters.labels, not just those observed in this trajectory
        # count_matrix = np.zeros([unique_labels.size, unique_labels.size])
        #
        # for frame in range(1, nT):  # start at frame 1. May need to truncate more as equilibration
        #     transitioned_from = self.clustered_state_sequence[:, frame - 1]
        #     transitioned_to = self.clustered_state_sequence[:, frame]
        #     for pair in zip(transitioned_from, transitioned_to):
        #         count_matrix[pair[0], pair[1]] += 1
        #
        # #transition_matrix = (count_matrix.T / count_matrix.sum(axis=1)).T
        #
        # A = np.zeros_like(self.converged_params['A'])[0, ..., :unique_labels.size]
        # sigma = np.zeros_like(self.converged_params['sigma'])[0, ..., :unique_labels.size]
        # A = []
        # sigma = []
        #
        # for l in unique_labels:
        #     ndx = [i for i, key in enumerate(map_states.keys()) if map_states[key] == l]
        #     A.append(self.converged_params['A'][..., ndx].mean(axis=0))
        #     sigma.append(self.converged_params['sigma'][..., ndx].mean(axis=0))
        #
        # # initial state distribution
        # pi_init = np.zeros([unique_labels.size])
        # for t in range(self.clustered_state_sequence.shape[0]):
        #     pi_init[self.clustered_state_sequence[t, 0]] += 1
        # pi_init /= pi_init.sum()  # normalize
        #
        # self.clustered_parameters = dict(A=A, sigma=sigma, pi_init=pi_init, count_matrix=count_matrix)

    def summarize_results(self, cmap=plt.cm.jet, traj_no=0, plot_dim='all', savename=None, shuffle=False,
                          crange=(50, 225)):
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

        print(np.diagonal(estimated_transition_matrix))
        # print('\nEstimated Transition Matrix:\n')
        # print(estimated_transition_matrix)

        if self.labels is not None:

            actual_transition_matrix = self.actual_T
            # rms = np.sqrt(np.square(estimated_transition_matrix - actual_transition_matrix).mean())
            #
            # print('\nActual Transition Matrix:\n')
            # print(actual_transition_matrix)
            #
            # print('\nRoot mean squared error between estimated and true matrices: %.4f\n' % rms)

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

        #colors = np.array([cmap(i) for i in np.random.choice(np.arange(cmap.N), size=self.max_states)])
        #shown_colors = np.array([cmap(i) for i in np.linspace(50, 225, len(found_states)).astype(int)])
        colors = np.array([cmap(i) for i in np.linspace(crange[0], crange[1], len(found_states)).astype(int)])

        if shuffle:
            np.random.shuffle(colors)

        #colors = np.array([cmap(i) for i in np.linspace(50, 225, np.max(found_states) + 1).astype(int)])
        #colors[found_states] = shown_colors

        # for setting custom colors
        # from matplotlib import colors as mcolors
        # colors = np.array([mcolors.to_rgba(i) for i in
        #                    ['xkcd:black', 'xkcd:orange', 'xkcd:red', 'xkcd:green', 'xkcd:gold', 'xkcd:violet',
        #                     'xkcd:yellow', 'xkcd:brown', 'xkcd:navy', 'xkcd:pink', 'xkcd:lavender', 'xkcd:magenta',
        #                    'xkcd:aqua', 'xkcd:silver', 'xkcd:purple', 'xkcd:blue']])
        # colors = np.array([mcolors.to_rgba(i) for i in ['blue', 'blue', 'blue']])
        # colors = np.array(['xkcd:blue', 'xkcd:orange', 'xkcd:gold', 'xkcd:red', 'xkcd:green', 'xkcd:magenta'])

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

                # y = self.com[(1 + self.order):, traj_no, i]
                y = self.trajectories[:, traj_no, i]
                # print(self.trajectories[:, traj_no, i].shape)
                #
                # print(nT)
                #ax_estimated[i].plot(np.arange(nT - 1) * self.dt / 1000, y)

                if dim > 1:
                    ax = ax_estimated[i]
                else:
                    ax = ax_estimated

                ax.add_collection(
                    multicolored_line_collection(np.arange(nT) * self.dt / 1000, y, z, colors))  # plot

                ax.set_xlim([0, nT * self.dt / 1000])
                ax.set_ylim([y.min(), y.max()])

                #ax_estimated[i].plot([0, 5000], [0, 0], '--', color='black', lw=2)
                #ax[1].plot([0, 5000], [0, 0], '--', color='black', lw=2)

        if self.labels is not None:
            ax_estimated.set_title('Estimated State Sequence', fontsize=16)
            ax_estimated.set_xlim([0, nT * self.dt])
            ymin = self.trajectories[:, traj_no, 0].min()
            if dim > 1:
                ymax = self.trajectories[:, traj_no, 1:].max() + (dim - 1) * shift
            else:
                ymax = self.trajectories[:, traj_no, 0].max()

            ax_estimated.set_ylim([ymin, ymax])
            # ax_estimated.set_ylim([self.com[:, traj_no, 2].min(), self.com[:, traj_no, 2].max()])

            ax_estimated.tick_params(labelsize=14)
            ax_estimated.set_xlabel('Time', fontsize=14)
        else:

            if dim > 1:
                ax1 = ax_estimated[0]
                ax2 = ax_estimated[-1]
            else:
                ax1 = ax_estimated
                ax2 = ax_estimated

            ax1.set_title('Estimated State Sequence', fontsize=16)
            ax2.set_xlabel('Time (ns)', fontsize=14)

            # if dim > 1:
            #     ax1 = ax_estimated[0]
            #     ax2 = ax_estimated[-1]
            # else:
            #     ax1 = ax_estimated
            #     ax2 = ax_estimated

            if dim == 2:
                ax1.set_ylabel('r-coordinate', fontsize=14)
                ax2.set_ylabel('z-coordinate', fontsize=14)
            elif dim == 3:
                ax_estimated[0].set_ylabel('x-coordinate', fontsize=14)
                ax_estimated[1].set_ylabel('y-coordinate', fontsize=14)
                ax_estimated[2].set_ylabel('z-coordinate', fontsize=14)

        plt.tick_params(labelsize=14)
        plt.tight_layout()

        if savename is not None:
            plt.savefig(savename)

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
            #dummy = sum(found_states)  # add all the found states to get a unique dummy state label
            dummy = 0
            while dummy in found_states:
                dummy += 1
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


def multicolored_line_collection(x, y, z, colors, lw=2):
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
    lc.set_linewidth(lw)

    return lc


class ClusteredParameters:

    def __init__(self, ihmm):

        print('hello')


def get_clustered_parameters(ihmm, clusters, order=1):
    """ Combine data from multiple trajectories estimate clustered state parameters

    :param ihmm: A single InfiniteHMM (not implemented) or a list containing instances of InfiniteHMM classes.
    :param clusters: Cluster object
    :param order: autoregressive order (only order 1 is implemented right now)

    :type InfiniteHMM instance or list of InfiniteHMM instances
    :type hdphmm.cluster.Cluster object
    :type order: int

    :return: dictionary of parameters
    """

    nclusters = clusters.nclusters

    # need A, sigma, transition matrix, pi_init

    A = [[] for n in range(nclusters)]
    sigma = [[] for n in range(nclusters)]
    count_matrix = np.zeros([nclusters, nclusters])
    pi_init = np.zeros([nclusters])
    for i, trajectory in enumerate(ihmm):  # loop through all of the trajectories
        params = trajectory.clustered_parameters
        for n in range(nclusters):
            if i == 0:
                A[n] = params['A'][n]
                sigma[n] = params['sigma'][n]
            else:
                A[n] = np.concatenate((A[n], params['A'][n]), axis=-1)
                sigma[n] = np.concatenate((sigma[n], params['sigma'][n]), axis=-1)

        count_matrix += params['count_matrix']
        pi_init += params['pi_init']

    A = np.array([a.mean(axis=-1) for a in A])
    sigma = np.array([s.mean(axis=-1) for s in sigma])

    transition_matrix = (count_matrix.T / count_matrix.sum(axis=1)).T
    pi_init = pi_init / pi_init.sum()

    return dict(A=A, sigma=sigma, transition_matrix=transition_matrix, pi_init=pi_init)
