#!/usr/bin/env python

from sklearn.metrics import silhouette_score
from LLC_Membranes.llclib import file_rw
import numpy as np
import matplotlib.pyplot as plt
from hdphmm.cluster import Cluster
import sys

try:
    res = sys.argv[1]
except IndexError:
    res = 'MET'

final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_diags_dtsigma0.25_dtA0.25.pl' %res)

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

fig, ax = plt.subplots(2, 2, figsize=(11, 9))
linkages = ['ward', 'average', 'complete', 'single']

for i, linkage in enumerate(linkages):

    silhouette_avg = []
    silhouette_avg_A = []
    silhouette_avg_sig = []
    silhouette_avg_mu = []

    nclust = np.arange(2, 15)

    for n in nclust:

        nclusters_sigma = n
        nclusters_A = n

        algorithm = 'agglomerative'

        #cluster = Cluster({'A': A, 'sigma': sigma}, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=n)
        
        cluster_means = Cluster({'mu': mu[0,:]}, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=n)

        #sig_cluster = Cluster(sig_params, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=nclusters_sigma, linkage=linkage)
        #A_cluster = Cluster(A_params, eigs=eigs, diags=diags, algorithm=algorithm, nclusters=nclusters_A, linkage=linkage)

        #cluster.fit()
        cluster_means.fit()
        #sig_cluster.fit()
        #A_cluster.fit()

        #silhouette_avg.append(silhouette_score(cluster.X, cluster.labels))
        #silhouette_avg_sig.append(silhouette_score(sig_cluster.X, sig_cluster.labels))
        #silhouette_avg_A.append(silhouette_score(A_cluster.X, A_cluster.labels))
        silhouette_avg_mu.append(silhouette_score(cluster_means.X, cluster_means.labels))

    #print(cluster.X)
    ax1 = i // 2
    ax2 = i % 2

    ax[ax1, ax2].set_title('%s' % linkage, fontsize=14)

    ax[ax1, ax2].plot(nclust, silhouette_avg_mu, lw=2, label='$\mu$')
    #ax[ax1, ax2].plot(nclust, silhouette_avg_sig, lw=2, label='sigma only')   
    #ax[ax1, ax2].plot(nclust, silhouette_avg_A, lw=2, label='A only') 
    #ax[ax1, ax2].plot(nclust, silhouette_avg, lw=2, label='A, Sigma')
    ax[ax1, ax2].set_xlabel('Number of clusters', fontsize=14)
    ax[ax1, ax2].set_ylabel('Silhouette Score', fontsize=14)
    ax[ax1, ax2].set_ylim(0, 1)
    ax[ax1, ax2].tick_params(labelsize=14)
    ax[ax1, ax2].legend(fontsize=14)

plt.tight_layout()
plt.show()
