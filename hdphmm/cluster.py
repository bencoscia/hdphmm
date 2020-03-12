#!/usr/bin/env python

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
import numpy as np


class Cluster:

    def __init__(self, params, distance_threshold=2., eigs=False, means=False, algorithm='bayesian', ncomponents=10, max_iter=1500,
                 weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None,
                 mean_precision_prior=1, init_params='random'):
        """ Cluster like parameter sets

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

        self.distance_threshold = distance_threshold
        self.means = means
        self.ncomponents = ncomponents
        self.max_iter = max_iter
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.gamma = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.init_params = init_params
        self.eigs = eigs
        self.nclusters = None

        #print(params['mu'])

        if isinstance(params, dict):

            A = self._flatten_A(params['A'])
            sigma = self._flatten_sigma(params['sigma'])

        elif isinstance(params, list):

            A = self._flatten_A(params[0]['A'])
            sigma = self._flatten_sigma(params[0]['sigma'])

            for param_set in params[1:]:

                A = np.concatenate((A, self._flatten_A(param_set['A'])))
                sigma = np.concatenate((sigma, self._flatten_sigma(param_set['sigma'])))

        else:

            raise Exception('Input data type not recognized. Please pass a list or a dict.')

        if len(A.shape) < 2:
            A = A[:, np.newaxis]
            sigma = sigma[:, np.newaxis]

        self.X = np.concatenate((A, sigma), axis=1)

        self.X -= self.X.min(axis=0)
        self.X /= self.X.max(axis=0)

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

         self.clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold)

    def _bayesian(self):

        self.clusters = BayesianGaussianMixture(n_components=self.ncomponents, max_iter=self.max_iter,
                                                weight_concentration_prior_type=self.weight_concentration_prior_type,
                                                weight_concentration_prior=self.gamma,
                                                mean_precision_prior=self.mean_precision_prior,
                                                init_params=self.init_params, verbose=1)

    def _flatten_A(self, A):

        if self.eigs:

            reordered = np.moveaxis(A, -1, 2)[:, 0, ...]  # reorder axes of A
            eigs = np.linalg.eig(reordered)[0]  # eigenvalues of each matrix
            a = eigs.max(axis=2).real  # get max eigenvalue for each matrix

        else:

            a = np.zeros([A.shape[0], A.shape[-1], A.shape[2]*A.shape[3]])
            for i in range(A.shape[-1]):
                for j in range(A.shape[0]):
                    a[j, i, :] = A[j, 0, ..., i].flatten()

        if self.means:

            return a.mean(axis=0)

        else:

            return np.reshape(a, (a.shape[0]*a.shape[1], a.shape[2]))

    def _flatten_sigma(self, sigma):

        if self.eigs:

            reordered = np.moveaxis(sigma, -1, 1)  # reorder axes for use with np.linalg.eig
            eigs = np.linalg.eig(reordered)[0]  # eigenvalues of each covariance matrix
            sig = eigs.max(axis=2).real

        else:

            sig = np.zeros([sigma.shape[0], sigma.shape[-1], sigma.shape[1]*sigma.shape[2]])
            for i in range(sigma.shape[-1]):
                for j in range(sigma.shape[0]):
                    w, v = np.linalg.eig(sigma[j, ..., i])
                    sig[j, i, :] = v.flatten()

        if self.means:

            return sig.mean(axis=0)

        else:

            return np.reshape(sig, (sig.shape[0]*sig.shape[1], sig.shape[2]))

