#!/usr/bin/env python

from sklearn.metrics import silhouette_score
from LLC_Membranes.llclib import file_rw
import numpy as np
import matplotlib.pyplot as plt
from hdphmm.cluster import Cluster

final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_MET_diags_dtsigma0.25_dtA0.25.pl')

ihmmr = final_parameters['ihmmr']

A = None
sigma = None
mu = None

for t in range(24):

    estimated_states = ihmmr[t].z[0, :]
    found_states = list(np.unique(estimated_states))

    a = np.zeros([2, 2, len(found_states)])  # should probably an include a dimension for AR order
    s = np.zeros([2, 2, len(found_states)])
    m = np.zeros([2, len(found_states)])

    for i, state in enumerate(found_states):

        Amean = ihmmr[t].converged_params['A'][:, 0, ..., i].mean(axis=0)
        sigmamean = ihmmr[t].converged_params['sigma'][:, ..., i].mean(axis=0)

        # we want to cluster on unconditional mean
        mucond = ihmmr[t].converged_params['mu'][..., i].mean(axis=0)  # conditional mean
        mumean = np.linalg.inv(np.eye(2) - Amean) @ mucond # unconditional mean

        a[..., i] = Amean
        s[..., i] = sigmamean
        m[:, i] = mumean

    if A is None:
        A = a
        sigma = s
        mu = m
    else:
        A = np.concatenate((A, a), axis=-1)
        sigma = np.concatenate((sigma, s), axis=-1)
        mu = np.concatenate((mu, m), axis=-1)

A_params = {'A': A}
sig_params = {'sigma': sigma}

eigs = False
diags = True

silhouette_avg = []
silhouette_avg_A = []
silhouette_avg_sig = []

nclust = np.arange(2, 50)

for n in nclust:

    nclusters_sigma = n
    nclusters_A = n

    algorithm = 'agglomerative'

    cluster = Cluster({'A': A, 'sigma': sigma}, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=n)

    sig_cluster = Cluster(sig_params, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=nclusters_sigma)
    A_cluster = Cluster(A_params, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=nclusters_A)

    cluster.fit()
    sig_cluster.fit()
    A_cluster.fit()

    silhouette_avg.append(silhouette_score(cluster.X, cluster.labels))
    silhouette_avg_sig.append(silhouette_score(sig_cluster.X, sig_cluster.labels))
    silhouette_avg_A.append(silhouette_score(A_cluster.X, A_cluster.labels))

#print(cluster.X)
plt.plot(nclust, silhouette_avg_sig, lw=2, label='sigma only')   
plt.plot(nclust, silhouette_avg_A, lw=2, label='A only') 
plt.plot(nclust, silhouette_avg, lw=2, label='A, Sigma')
plt.xlabel('Number of clusters', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.ylim(0, 1)
plt.tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
