#!/usr/bin/env python

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from hdphmm.utils import stats


class Cluster:

    def __init__(self, params, distance_threshold=2., eigs=False, diags=False, means=False, algorithm='bayesian', ncomponents=10, max_iter=1500,
                 weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None,
                 mean_precision_prior=1, init_params='random', nclusters=None):
        """ Cluster like parameter sets

        NOTE this is only made for AR(1) at the moment.

        :param params: parameters for each trajectory. If a single trajectory was analyzed, you can just pass a single \
        dict, otherwise pass a list of dicts
        :param distance_threshold: clustering parameter
        :param eigs: if True, use largest eigenvalue of AR coefficient and covariance matrices instead of flattened \
        matrices.
        :param means: if True, take the mean of each set of parameters
        :param algorithm: type of clustering algorithm.'bayesian' and 'agglomerative' are implemented

        :type params: list or dict
        :type distance_threshold: float
        :type eigs: bool
        :type algorithm: str
        """

        cluster_fxn = {'bayesian': self._bayesian, 'agglomerative': self._agglomerative}

        try:
            self.cluster_fxn = cluster_fxn[algorithm]
        except KeyError:
            raise Exception("Clustering algorithm, '%s', not implemented. Use either 'bayesian' or 'agglomerative'" % algorithm)

        self.means = means
        self.ncomponents = ncomponents
        self.max_iter = max_iter
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.gamma = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.init_params = init_params
        self.eigs = eigs
        self.diags = diags
        self.nclusters = nclusters
        if self.nclusters is not None:
            self.distance_threshold = None
        else:
            self.distance_threshold = distance_threshold

        if isinstance(params, dict):

            A = None
            sigma = None

            if 'A' in params.keys():
                A = self._flatten_A(params['A'])

            if 'sigma' in params.keys():
                sigma = self._flatten_sigma(params['sigma'])

            mu = None
            if 'mu' in params.keys():

                mu = params['mu']

                if len(mu.shape) == 1:
                    mu = mu[:, np.newaxis]

            T = None
            if 'T' in params.keys():

                T = params['T']

                if len(T.shape) == 1:
                    T = T[:, np.newaxis]

        elif isinstance(params, list):

            A = self._flatten_A(params[0]['A'])
            sigma = self._flatten_sigma(params[0]['sigma'])

            for param_set in params[1:]:

                A = np.concatenate((A, self._flatten_A(param_set['A'])))
                sigma = np.concatenate((sigma, self._flatten_sigma(param_set['sigma'])))

        else:

            raise Exception('Input data type not recognized. Please pass a list or a dict.')

        if A is not None:
            if len(A.shape) < 2:
                A = A[:, np.newaxis]

            if sigma is not None:

                if len(sigma.shape) < 2:
                    sigma = sigma[:, np.newaxis]

                self.X = np.concatenate((A, sigma), axis=1)
            else:
                self.X = A
        elif sigma is not None:

            if len(sigma.shape) < 2:
                sigma = sigma[:, np.newaxis]

            self.X = sigma

        if mu is not None:
            if A is None and sigma is None:
                self.X = mu
            else:
                self.X = np.concatenate((self.X, mu), axis=1)

        if T is not None:
            if A is None and sigma is None and mu is None:
                self.X = T
            else:
                self.X = np.concatenate((self.X, T), axis=1)

        if algorithm is 'agglomerative':

            for d in range(self.X.shape[1]):

                outliers_removed = stats.remove_outliers(self.X[:, d])
                # self.X[:, d] -= outliers_removed.min()
                # self.X[:, d] /= outliers_removed.max()

                self.X[:, d] -= outliers_removed.mean()
                self.X[:, d] /= outliers_removed.std()

        self.labels = None
        self.clusters = None

        # clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        # self.labels = clusters.fit_predict(X)

    def fit(self):

        self.cluster_fxn()
        self.labels = self.clusters.fit_predict(self.X)
        self._remap_labels()

    def _remap_labels(self):
        """ relabel the labels counting from zero

        :return:
        """

        map_states = dict()
        unique_labels = np.unique(self.labels)
        for i, label in enumerate(unique_labels):
            map_states[label] = i

        self.labels = [map_states[l] for l in self.labels]
        self.nclusters = unique_labels.size

    def _agglomerative(self):

         self.clusters = AgglomerativeClustering(n_clusters=self.nclusters, distance_threshold=self.distance_threshold,
                                                 linkage='ward')

    def _bayesian(self):

        self.clusters = BayesianGaussianMixture(n_components=self.ncomponents, max_iter=self.max_iter,
                                                weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                weight_concentration_prior=self.gamma,
                                                mean_precision_prior=self.mean_precision_prior,
                                                init_params=self.init_params, verbose=0)

    def _flatten_A(self, A):

        if self.eigs:

            reordered = np.moveaxis(A, -1, 0)#[:, 0, ...]  # reorder axes of A
            eigs = np.linalg.eig(reordered)[0]  # eigenvalues of each matrix

            return eigs.real  # imaginary component usually very close to zero

        elif self.diags:

            return np.array([np.diag(A[..., a]) for a in range(A.shape[2])])

        else:

            # a = np.zeros([A.shape[0], A.shape[-1], A.shape[2]*A.shape[3]])
            # for i in range(A.shape[-1]):
            #     for j in range(A.shape[0]):
            #         a[j, i, :] = A[j, 0, ..., i].flatten()

            return A.reshape((A.shape[0]*A.shape[1], A.shape[2])).T

        # if self.means:
        #
        #     return a.mean(axis=0)
        #
        # else:
        #
        #     return np.reshape(a, (a.shape[0]*a.shape[1], a.shape[2]))

    def _flatten_sigma(self, sigma):

        if self.eigs:

            reordered = np.moveaxis(sigma, -1, 0)  # reorder axes for use with np.linalg.eig
            eigs = np.linalg.eig(reordered)[0]  # eigenvalues of each covariance matrix

            return eigs.real

        elif self.diags:

            return np.array([np.diag(sigma[..., s]) for s in range(sigma.shape[2])])

        else:

            # This will need to modified for high AR orders
            # can edit this to get rid of redundant symmetric covariance terms
            return sigma.reshape((sigma.shape[0]*sigma.shape[1], sigma.shape[2])).T

        #     sig = np.zeros([sigma.shape[0], sigma.shape[-1], sigma.shape[1]*sigma.shape[2]])
        #     for i in range(sigma.shape[-1]):
        #         for j in range(sigma.shape[0]):
        #             w, v = np.linalg.eig(sigma[j, ..., i])
        #             sig[j, i, :] = v.flatten()
        #
        # if self.means:
        #
        #     return sig.mean(axis=0)
        #
        # else:
        #
        #     return np.reshape(sig, (sig.shape[0]*sig.shape[1], sig.shape[2]))

