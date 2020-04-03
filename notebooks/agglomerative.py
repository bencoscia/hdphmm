#!/usr/bin/env python

import numpy as np
import hdphmm
import mdtraj
import matplotlib.pyplot as plt
from hdphmm.cluster import Cluster
from LLC_Membranes.llclib import file_rw
from hdphmm.generate_timeseries import GenARData
from hdphmm import timeseries as ts
from matplotlib.patches import Circle
import sys


def ihmm(res, niter=100, cluster=1, dt_A=1, dt_sigma=0.25, algorithm='agglomerative', final_parameters=None, nclusters_sigma=None, nclusters_A=None, tot_clusters=None, combine_clusters=False):

    cluster_variables = ['diags', 'eigs']
    cluster_vars = cluster_variables[cluster]

    difference = False  # take first order difference of solute trajectories
    observation_model='AR'  # assume an autoregressive model (that's the only model implemented)
    order = 1  # autoregressive order
    max_states = 100  # More is usually better
    traj_no = np.arange(24).tolist() # np.arange(10).tolist()# [2]# None # np.arange(24)#2
    dim = [0, 1, 2]  # dimensions of trajectory to keep
    prior = 'MNIW-N'  # MNIW-N (includes means) or MNIW (forces means to zero)
    keep_xy = True
    save_every = 1

    # You can define a dictionary with some spline paramters
    spline_params = {'npts_spline': 10, 'save': True, 'savename': 'trajectories/spline_%s.pl' % res}

    com_savename = 'trajectories/com_xy_radial_%s.pl' % res

    com = 'trajectories/com_xy_radial_%s.pl' % res # center of mass trajectories. If it exists, we can skip loading the MD trajectory and just load this

    if final_parameters is None:

        # We will be applying the IHMM to each tr|ajectory independently
        ihmm = [[] for i in range(24)] 

        # initialize the ihmm for each trajectory
        for t in traj_no:
    
            ihmm[t] = hdphmm.InfiniteHMM(com, traj_no=t, load_com=True, difference=difference,
                                 observation_model=observation_model, order=order, max_states=max_states,
                                 dim=dim, spline_params=spline_params, prior=prior,
                                 hyperparams=None, keep_xy=keep_xy, com_savename=com_savename,
                                 radial=True, save_com=False, save_every=save_every, res=res)

        for i in traj_no:
            ihmm[i].inference(niter)

        for i in traj_no:
            ihmm[i]._get_params(quiet=True)

        ihmmr = [[] for i in traj_no]

        # convert to radial
        for i in traj_no:

            radial = np.zeros([ihmm[i].com.shape[0], 1, 2])
            radial[:, 0, 0] = np.linalg.norm(ihmm[i].com[:, 0, :2], axis=1)
            radial[:, 0, 1] = ihmm[i].com[:, 0, 2]

            ihmmr[i] = hdphmm.InfiniteHMM((radial, ihmm[i].dt), traj_no=[0], load_com=False, difference=False,
                                   order=1, max_states=100,
                                   dim=[0, 1], spline_params=spline_params, prior='MNIW-N',
                                   hyperparams=None, save_com=False, state_sequence=ihmm[i].z)

            ihmmr[i].inference(niter)

        for i in traj_no:
            ihmmr[i]._get_params(traj_no=0)

    else:

        ihmm = final_parameters['ihmm']
        ihmmr = final_parameters['ihmmr']

    # Cluster radial params

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

    # default is diags
    eigs = False
    diags = True
    if cluster_vars == 'eigs':
        eigs = True
        diags = False

    if combine_clusters:

        params = {'sigma': sigma, 'A': A}
        sig_cluster = Cluster(params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=None, nclusters=tot_clusters)        
        sig_cluster.fit()

        new_labels = sig_cluster.labels

        print('Found %d clusters' % np.unique(sig_cluster.labels).size)
        
    else:

        sig_params = {'sigma': sigma}
        A_params = {'A': A}

        sig_cluster = Cluster(sig_params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=dt_sigma, nclusters=nclusters_sigma)
        A_cluster = Cluster(A_params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=dt_A, nclusters=nclusters_A)

        sig_cluster.fit()
        A_cluster.fit()

        nA_clusters = np.unique(A_cluster.labels).size
        nsig_clusters = np.unique(sig_cluster.labels).size
        print('Found %d sigma clusters' % nsig_clusters)
        print('Found %d A clusters' % nA_clusters)

        cluster_matrix = np.zeros([nA_clusters, nsig_clusters])

        new_clusters = np.zeros([A.shape[-1]])

        for state in range(A.shape[-1]):
            new_clusters[state] = A_cluster.labels[state] * nsig_clusters + sig_cluster.labels[state]
    
        print('Found %d total clusters' % np.unique(new_clusters).size)

        all_labels = np.unique(new_clusters).astype(int)

        new_label_dict = {l:i for i, l in enumerate(all_labels)}

        new_labels = [new_label_dict[int(i)] for i in new_clusters]

        sig_cluster.labels = new_labels

    all_state_params = {'A': A, 'sigma': sigma, 'mu': mu, 'state_labels': new_labels}

    ndx = 0
    for i in traj_no:
        end = ndx + len(ihmmr[i].found_states)
        labels = new_labels[ndx:end]
        ndx = end
        ihmmr[i].reassign_state_sequence(sig_cluster, labels=labels)

    all_mu = None
    for t in traj_no:

        m = ihmmr[t].converged_params['mu'].mean(axis=0)
        phi = ihmmr[t].converged_params['A'][:, 0, ..., :].mean(axis=0)

        # convert to unconditional mean
        for i in range(m.shape[1]):
            m[:, i] = np.linalg.inv(np.eye(2) - phi[..., i]) @ m[:, i]

        if all_mu is None:
            all_mu = m
        else:
            all_mu = np.concatenate((all_mu, m), axis=1)

    nclusters = np.unique(sig_cluster.labels).size
    mu = np.zeros([nclusters, 2])
    for i in range(nclusters):
        ndx = np.where(np.array(sig_cluster.labels) == i)[0]
        mu[i, :] = all_mu[:, ndx].mean(axis=1)

    mean_zero = []

    for t in traj_no:

        zeroed = ihmmr[t].subtract_mean(traj_no=0, simple_mean=True)
        mean_zero.append(zeroed)

    mean_zero = np.array(mean_zero)

    z = None
    for t in traj_no:

        seq = ihmmr[t].clustered_state_sequence[:, :]
        if z is None:
            z = seq
        else:
            z = np.concatenate((z, seq), axis=0)

    ihmm_final = hdphmm.InfiniteHMM((np.moveaxis(mean_zero, 0, 1), ihmmr[t].dt), traj_no=None, load_com=False, difference=False,
                                   order=1, max_states=mu.shape[0],
                                   dim=[0, 1], spline_params=spline_params, prior='MNIW',
                                   hyperparams=None, save_com=False, state_sequence=z[:, 1:])

    ihmm_final.inference(niter)

    nclusters = np.unique(z).size

    ntraj = len(traj_no)

    A = np.zeros([ntraj, nclusters, 2, 2])
    sigma = np.zeros_like(A)
    weights = np.zeros([ntraj, nclusters])

    for t in range(len(traj_no)):
        ihmm_final._get_params(traj_no=t, quiet=True)
        for i, ndx in enumerate(ihmm_final.found_states):
            A[t, ndx, ...] = ihmm_final.converged_params['A'][:, 0, ..., i].mean(axis=0)
            sigma[t, ndx, ...] = ihmm_final.converged_params['sigma'][:, ..., i].mean(axis=0)
            weights[t, ndx] = np.where(ihmm_final.z[t, :] == ndx)[0].size

    A_final = np.zeros([nclusters, 1, 2, 2])
    sigma_final = np.zeros([nclusters, 2, 2])
    for c in range(nclusters):
        if weights[:, c].sum() > 0:
            A_final[c, 0, ...] = np.average(A[:, c, ...], axis=0, weights=weights[:, c])
            sigma_final[c, ...] = np.average(sigma[:, c, ...], axis=0, weights=weights[:, c])


    m = np.zeros_like(mu)
    for i in range(m.shape[0]):
        m[i, :] = (np.eye(2) - A_final[i, 0, ...]) @ mu[i, :]

    found_states = np.unique(ihmm_final.z)
    ndx_dict = {found_states[i]: i for i in range(len(found_states))}

    count_matrix = np.zeros([nclusters, nclusters])

    nT = ihmm_final.nT
    for frame in range(1, nT - 1):  # start at frame 1. May need to truncate more as equilibration
        transitioned_from = [ndx_dict[i] for i in ihmm_final.z[:, frame - 1]]
        transitioned_to = [ndx_dict[i] for i in ihmm_final.z[:, frame]]
        for pair in zip(transitioned_from, transitioned_to):
            count_matrix[pair[0], pair[1]] += 1

    # The following is very similar to ihmm3.pi_z. The difference is due to the dirichlet process.
    transition_matrix = (count_matrix.T / count_matrix.sum(axis=1)).T

    init_state = ihmm_final.z[:, 0]
    pi_init = np.zeros([nclusters])
    for i, c in enumerate(ihmm_final.found_states):
        pi_init[i] = np.where(init_state == c)[0].size

    pi_init /= pi_init.sum()

    final_parameters = {'A': A_final, 'sigma': sigma_final, 'mu': mu, 'T': transition_matrix, 'pi_init': pi_init, 'z': ihmm_final.z, 'ihmmr': ihmmr, 'ihmm':ihmm, 'all_state_params': all_state_params}

    if combine_clusters:

        file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_agglomerative_%s_%s_combined_%d.pl' % (res, cluster_vars, tot_clusters)) 

    else:

        if nclusters_A is None:

            file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_agglomerative_%s_%s_dtsigma%.2f_dtA%.2f.pl' %(res, cluster_vars, dt_sigma, dt_A))

        else:

            file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_agglomerative_%s_%s_nsigma%d_nA%d.pl' %(res, cluster_vars, nclusters_sigma, nclusters_A))

    return final_parameters


def msd(final_parameters, res):

    nboot = 200
    frac = 0.4
    ntraj = 100
    nsteps = 4806
    dt = 0.5
    endshow=2000

    trajectory_generator = GenARData(params=final_parameters)
    trajectory_generator.gen_trajectory(nsteps, ntraj, bound_dimensions=[0])

    # Calculate MSD and plot
    msd = ts.msd(trajectory_generator.traj, 1)
    error = ts.bootstrap_msd(msd, nboot, confidence=68)

    t = np.arange(endshow)*dt
    plt.plot(t, msd.mean(axis=1)[:endshow], lw=2, color='xkcd:blue')
    plt.fill_between(t, msd.mean(axis=1)[:endshow] + error[0, :endshow], msd.mean(axis=1)[:endshow] - error[1, :endshow], alpha=0.3, color='xkcd:blue')

    MD_MSD = file_rw.load_object('trajectories/%s_msd.pl' % res)
    plt.title(names[res], fontsize=18)
    plt.plot(t, MD_MSD.MSD_average[:endshow], color='black', lw=2)
    plt.fill_between(t, MD_MSD.MSD_average[:endshow] + MD_MSD.limits[0, :endshow], MD_MSD.MSD_average[:endshow] - MD_MSD.limits[1, :endshow], alpha=0.3, color='black')
    plt.tick_params(labelsize=14)
    plt.xlabel('Time (ns)', fontsize=14)
    plt.ylabel('Mean Squared Displacement (nm$^2$)', fontsize=14)

    plt.show()


def plot_realization(final_parameters, res):

    MD_MSD = file_rw.load_object('trajectories/%s_msd.pl' % res)

    nsteps = MD_MSD.MSD.shape[0]

    trajectory_generator = GenARData(params=final_parameters)
    trajectory_generator.gen_trajectory(nsteps, 1, bound_dimensions=[0])

    fig, ax = plt.subplots(2, 1, figsize=(12, 5))

    ax[0].plot(trajectory_generator.traj[:, 0, 1], lw=2)
    ax[1].plot(trajectory_generator.traj[:, 0, 0], lw=2)

    ax[0].set_xlabel('Step number', fontsize=14)
    ax[0].set_ylabel('z coordinate', fontsize=14)
    ax[0].tick_params(labelsize=14)

    ax[1].set_xlabel('Step number', fontsize=14)
    ax[1].set_ylabel('r coordinate', fontsize=14)
    ax[1].tick_params(labelsize=14)
    
    plt.show()

def view_clusters(final_parameters, shift=3, show_states='all'):

    all_params = final_parameters['all_state_params']
    A = all_params['A']
    sigma = all_params['sigma']
    new_clusters = all_params['state_labels']

    centroids = np.zeros([2, 3, np.unique(new_clusters).size])  # [A/sig, 2 dimensions + mass, cluster]

    if show_states == 'all':
        show_states = np.arange(np.unique(new_clusters).size)

    for i, c in enumerate(np.unique(new_clusters)):


        ndx = np.where(new_clusters == c)[0]

        Adiags = np.array([np.diag(A[..., a]) for a in ndx])
        sigdiags = np.array([np.diag(sigma[..., s]) for s in ndx])

        centroids[0, :, i] = np.concatenate((Adiags.mean(axis=0), [len(ndx)]))
        centroids[1, :, i] = np.concatenate((sigdiags.mean(axis=0), [len(ndx)]))
        
        if i in show_states:

            print('Cluster %d' % c)
            print(len(ndx))

            fig, ax = plt.subplots(2, 2, figsize=(12, 7))#, sharey=True)

            ax[1, 0].scatter(Adiags[:, 0], Adiags[:, 1])
            ax[1, 1].scatter(sigdiags[:, 0], sigdiags[:, 1])

            for j, n in enumerate(np.random.choice(ndx, size=min(4, len(ndx)), replace=False)):
                print(np.diag(A[..., n]), np.diag(sigma[..., n]))
                parameters = {'pi_init': [1], 'T': np.array([[1]]), 'mu': np.zeros(2), 'A': A[..., n][np.newaxis, np.newaxis, ...], 'sigma': sigma[..., n][np.newaxis, ...]}

                trajectory_generator = GenARData(params=parameters)
                trajectory_generator.gen_trajectory(1000, 1, state_no=0, progress=False)

                t = trajectory_generator.traj[:, 0, :]
                t -= t.mean(axis=0)

                ax[0, 0].plot(t[:, 1] + j*shift, lw=2)
                ax[0, 1].plot(t[:, 0] + j*shift, lw=2)

            ax[0, 0].set_title('$z$ direction', fontsize=16)
            ax[0, 1].set_title('$r$ direction', fontsize=16)

            ax[0, 0].set_xlabel('Step Number', fontsize=14)

            ax[0, 0].set_ylabel('Shifted $z$ coordinate', fontsize=14)
            ax[0, 1].set_xlabel('Step Number', fontsize=14)
            ax[0, 1].set_ylabel('Shifted $r$ coordinate', fontsize=14)

            ax[1, 0].set_xlabel('A(0, 0)', fontsize=14)
            ax[1, 0].set_ylabel('A(1, 1)', fontsize=14)
            ax[1, 1].set_xlabel('$\Sigma$(0, 0)', fontsize=14)
            ax[1, 1].set_ylabel('$\Sigma$(1, 1)', fontsize=14)

            ax[0, 0].tick_params(labelsize=14)
            ax[0, 1].tick_params(labelsize=14)
            ax[0, 0].tick_params(labelsize=14)
            ax[0, 1].tick_params(labelsize=14)

            plt.tight_layout()

            plt.show()

    # plot centroids
    bubble_scale = 0.02
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cmap = plt.cm.jet
    colors = np.array([cmap(i) for i in np.random.choice(np.arange(cmap.N), size=centroids.shape[-1])])
    for i in range(centroids.shape[-1]):
        
        if centroids[0, 2, i] > 5:

            ax[0].scatter(centroids[0, 0, i], centroids[0, 1, i], color=colors[i])
            ax[1].scatter(centroids[1, 0, i], centroids[1, 1, i], color=colors[i])
            ax[0].add_patch(Circle(centroids[0, :, i], np.sqrt(centroids[0, 2, i] / np.pi) * bubble_scale, color=colors[i], alpha=0.3, lw=2))
            ax[1].add_patch(Circle(centroids[1, :, i], np.sqrt(centroids[0, 2, i] / np.pi) * bubble_scale*0.4, color=colors[i], alpha=0.3, lw=2))

    ax[0].set_xlabel('A(0, 0)', fontsize=14)
    ax[0].set_ylabel('A(1, 1)', fontsize=14)
    ax[1].set_xlabel('$\Sigma$(0, 0)', fontsize=14)
    ax[1].set_ylabel('$\Sigma$(1, 1)', fontsize=14)
 
    ax[0].set_xlim(-.2, 1)
    ax[0].set_ylim(-.2, 1)

    ax[1].set_xlim(-0.1, 0.3)
    ax[1].set_ylim(-0.1, 0.3)

    ax[0].set_aspect(1)
    ax[1].set_aspect(1)
 
    plt.tight_layout()
    plt.show()


def cluster_behavior(params, top_percent):

    z = params['z']
    count = np.zeros(np.unique(z).size)
    
    for i in range(z.shape[0]):
        for j in range(count.size):
            count[j] += len(np.where(z[i, :] == j)[0])

    count /= count.sum()

    sorted_ndx = np.argsort(count)[::-1]

    stop = np.where(np.cumsum(count[sorted_ndx]) > (top_percent / 100))[0][0]
    prevelant_states = sorted_ndx[:(stop + 1)]

    shift = 0.75

    fig, ax = plt.subplots(1, 3, figsize=(12, 7), sharey=True, gridspec_kw={'width_ratios': [1, 1, 0.2]})

    trajectory_generator = GenARData(params=final_parameters)
    
    A = final_parameters['A']
    sigma = final_parameters['sigma']

    for i, s in enumerate(prevelant_states):
        print(np.diag(A[s, 0, ...]), np.diag(sigma[s, ...]))
        trajectory_generator.gen_trajectory(500, 1, state_no=s, progress=False)

        t = trajectory_generator.traj[:, 0, :]
        t -= t.mean(axis=0)

        ax[0].plot(t[:, 1] + i*shift, lw=2)
        ax[1].plot(t[:, 0] + i*shift, lw=2)
        ax[2].text(0, i*shift, '%.1f %%' % (100*count[s]), fontsize=20, horizontalalignment='center')
    
    ax[0].set_xlabel('Step Number', fontsize=14)
    ax[1].set_xlabel('Step Number', fontsize=14)
    ax[0].set_title('$z$ direction', fontsize=16)
    ax[1].set_title('$r$ direction', fontsize=16)
    ax[0].tick_params(labelsize=14)
    ax[1].tick_params(labelsize=14)
    ax[2].axis('off')
    plt.tight_layout()

    plt.show()


def test_cluster(final_parameters, dt_A=None, dt_sigma=None, nclusters_A=None, nclusters_sigma=None, show=True):

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

    print(A.shape)
    sig_params = {'sigma': sigma}
    A_params = {'A': A}

    # default is diags
    eigs = False
    diags = True
    if cluster_vars == 'eigs':
        eigs = True
        diags = False

    fig, ax = plt.subplots(1, 2)

    sig_cluster = Cluster(sig_params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=dt_sigma, nclusters=nclusters_sigma)
    A_cluster = Cluster(A_params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=dt_A, nclusters=nclusters_A)

    sig_cluster.fit()
    A_cluster.fit()

    sigma_clusters = np.zeros([np.unique(sig_cluster.labels).size, 2])
    for j, k in enumerate(np.unique(sig_cluster.labels)):
        #print('Cluster %d' % k)
        ndx = np.where(sig_cluster.labels == k)[0]
        diagonals = np.zeros([2, len(ndx)])
        for i, n in enumerate(ndx):
            diagonals[:, i] = np.diag(sigma[..., n])
        sigma_clusters[j, :] = diagonals.mean(axis=1)

    A_clusters = np.zeros([np.unique(A_cluster.labels).size, 2])
    for j, k in enumerate(np.unique(A_cluster.labels)):
        #print('Cluster %d' % k)
        ndx = np.where(A_cluster.labels == k)[0]
        diagonals = np.zeros([2, len(ndx)])
        for i, n in enumerate(ndx):
            diagonals[:, i] = np.diag(A[..., n])
        A_clusters[j, :] = diagonals.mean(axis=1)

    ax[0].scatter(A_clusters[:, 0], A_clusters[:, 1])
    ax[1].scatter(sigma_clusters[:, 0], sigma_clusters[:, 1])
   
    nA_clusters = np.unique(A_cluster.labels).size
    nsig_clusters = np.unique(sig_cluster.labels).size
    print('Found %d sigma clusters' % nsig_clusters)
    print('Found %d A clusters' % nA_clusters)

    cluster_matrix = np.zeros([nA_clusters, nsig_clusters])

    new_clusters = np.zeros([A.shape[-1]])

    for state in range(A.shape[-1]):
        new_clusters[state] = A_cluster.labels[state] * nsig_clusters + sig_cluster.labels[state]

    print('Found %d total clusters' % np.unique(new_clusters).size)

    if show:
        plt.show()
        shift = 3

        for ndx, c in enumerate(np.unique(new_clusters)):

            print('Cluster %d' % c)
    
            fig, ax = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
            fig2, ax2 = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    
            ndx = np.where(new_clusters == c)[0]
    
            print(len(ndx))

            Adiags = np.array([np.diag(A[..., a]) for a in ndx])
            sigdiags = np.array([np.diag(sigma[..., s]) for s in ndx])
    
            ax2[0].scatter(Adiags[:, 0], Adiags[:, 1])
            ax2[1].scatter(sigdiags[:, 0], sigdiags[:, 1])
    
            for i, n in enumerate(np.random.choice(ndx, size=min(4, len(ndx)), replace=False)):
                print(np.diag(A[..., n]), np.diag(sigma[..., n]))
        
                parameters = {'pi_init': [1], 'T': np.array([[1]]), 'mu': np.zeros(2), 'A': A[..., n][np.newaxis, np.newaxis, ...], 'sigma': sigma[..., n][np.newaxis, ...]}

                trajectory_generator = GenARData(params=parameters)
                trajectory_generator.gen_trajectory(1000, 1, state_no=0, progress=False)

                t = trajectory_generator.traj[:, 0, :]
                t -= t.mean(axis=0)

                ax[0].plot(t[:, 1] + i*shift, lw=2)
                ax[1].plot(t[:, 0] + i*shift, lw=2)
                #ax[2].text(0, s*shift, '%.1f %%' % (100*fraction[s]), fontsize=20, horizontalalignment='center')
    
            ax[0].set_xlabel('Step Number', fontsize=14)
            ax[1].set_xlabel('Step Number', fontsize=14)
            ax[0].set_title('$z$ direction', fontsize=16)
            ax[1].set_title('$r$ direction', fontsize=16)
            ax[0].tick_params(labelsize=14)
            ax[1].tick_params(labelsize=14)
            plt.tight_layout()

            plt.show()

load=True
view_clusters_= False
show_states = []
calculate_msd=True
plot_realization_=True
view_cluster_behavior=False
top_percent = 99
recluster_= True
test_clusters_ = False #True

# This determines what file will be loaded
combine_clusters = False
tot_clusters = 6

# Reclustering parameters
combine_clusters_new = False
new_tot_clusters = 30

# if these are None, the distance threshold will be used
# This will dictate what file is loaded
nclusters_A = 3
nclusters_sigma = 3

# for reclustering
new_nclusters_A = 6
new_nclusters_sigma = 7

res='MET'

# This will determine which file is loaded
dtA = 1.0 #.25 
dtsigma = 0.25

# This will define new parameters if recluster is True
new_dtA = 1
new_dtsigma = 0.25

algorithm='agglomerative'

names = {'MET': 'methanol', 'URE': 'urea', 'GCL': 'ethylene glycol', 'ACH': 'acetic acid'}

try:
    res = sys.argv[1]
except IndexError:
    res = 'MET'

try:
    cluster = int(sys.argv[2])
except IndexError:
    cluster=0

niter=100

print('Residue %s, cluster %d' % (res, cluster))
if load:

    cluster_variables = ['diags', 'eigs']
    cluster_vars = cluster_variables[cluster]

    if combine_clusters:

        final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_combined_%d.pl' % (res, cluster_vars, tot_clusters))

    else:

        if nclusters_A is None:

            final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_dtsigma%.2f_dtA%.2f.pl' % (res, cluster_vars, dtsigma, dtA))

        else:

            if nclusters_sigma is None:
                raise Exception('nclusters_sigma must not be None')

            final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_nsigma%d_nA%d.pl' % (res, cluster_vars, nclusters_sigma, nclusters_A))

else:

    final_parameters = ihmm(res, cluster=cluster, niter=niter, algorithm=algorithm, dt_A=dtA, dt_sigma=dtsigma, nclusters_sigma=nclusters_sigma, nclusters_A=nclusters_A, tot_clusters=tot_clusters,combine_clusters=combine_clusters)

if view_clusters_:

    view_clusters(final_parameters, show_states=show_states)

if view_cluster_behavior:
    cluster_behavior(final_parameters, top_percent=top_percent)

if recluster_:

    final_parameters = ihmm(res, cluster=cluster, niter=niter, algorithm=algorithm, dt_A=new_dtA, dt_sigma=new_dtsigma, final_parameters=final_parameters, nclusters_sigma=new_nclusters_sigma, nclusters_A=new_nclusters_A, tot_clusters=new_tot_clusters, combine_clusters=combine_clusters_new)

if test_clusters_:

    if nclusters_A is None:
        test_cluster(final_parameters, dt_A=dtA, dt_sigma=dtsigma)
    else:
        test_cluster(final_parameters, nclusters_sigma=nclusters_sigma, nclusters_A=nclusters_A)

if calculate_msd:
    msd(final_parameters, res)

if plot_realization_:

    plot_realization(final_parameters, res)
