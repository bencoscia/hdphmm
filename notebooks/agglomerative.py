#!/usr/bin/env python

import numpy as np
import hdphmm
import mdtraj
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from hdphmm.cluster import Cluster
from LLC_Membranes.llclib import file_rw
from hdphmm.generate_timeseries import GenARData
from hdphmm import timeseries as ts
from matplotlib.patches import Circle
import sys


def seed_sequence(com, traj_no, nseg=2, max_states=100, niter=10):

    segments = [[] for i in range(nseg)]
    pps = com[0].shape[0] // nseg  # point per segment
    for s in range(nseg):

        if s == 0:
            seg = (com[0][s*pps: (s + 1)*pps, [traj_no], :], com[1])
        else:
            seg = (com[0][s*pps - 1: (s + 1)*pps, [traj_no], :], com[1])

        segments[s] = hdphmm.InfiniteHMM(seg, traj_no=0, load_com=False, difference=False, 
                                 observation_model='AR', order=1, max_states=max_states,
                                 dim=[0, 1, 2], prior='MNIW-N', save_every=1, hyperparams=None)

    z = np.zeros([1, 0], dtype=int)
    for s in range(nseg):
        segments[s].inference(niter)
        zseg = segments[s].z + max_states * s
        z = np.concatenate((z, zseg), axis=1)

    new_labels = {x: i for i, x in enumerate(np.unique(z))}
    for i in range(z.shape[1]):
        z[0, i] = new_labels[z[0, i]]
        
    return z


def ihmm(res, niter=100, cluster=1, dt_A=1, dt_sigma=0.25, algorithm='agglomerative', final_parameters=None, nclusters_sigma=None, nclusters_A=None, tot_clusters=None, combine_clusters=False, nclusters_r=3, nclusters_T=5, order=1, seed=True):

    cluster_variables = ['diags', 'eigs']
    cluster_vars = cluster_variables[cluster]

    difference = False  # take first order difference of solute trajectories
    observation_model='AR'  # assume an autoregressive model (that's the only model implemented)
    order = order  # autoregressive order
    max_states = 200  # More is usually better
    traj_no = np.arange(24).tolist() # np.arange(10).tolist()# [2]# None # np.arange(24)#2
    dim = [0, 1, 2]  # dimensions of trajectory to keep
    prior = 'MNIW-N'  # MNIW-N (includes means) or MNIW (forces means to zero)
    keep_xy = True
    save_every = 10

    # You can define a dictionary with some spline paramters
    spline_params = {'npts_spline': 10, 'save': True, 'savename': 'trajectories/spline_%s.pl' % res}

    com_savename = 'trajectories/com_xy_radial_%s.pl' % res

    com = 'trajectories/com_xy_radial_%s.pl' % res # center of mass trajectories. If it exists, we can skip loading the MD trajectory and just load this
    com_raw = file_rw.load_object(com)
    if final_parameters is None:

        # We will be applying the IHMM to each tr|ajectory independently
        ihmm = [[] for i in range(24)] 

        # initialize the ihmm for each trajectory
        for t in traj_no:

            if seed:

                z = seed_sequence(com_raw, t, nseg=4, max_states=max_states, niter=3)
                print('Seeding with %d states' % np.unique(z).size)

            else:

                z = None
    
            ihmm[t] = hdphmm.InfiniteHMM(com, traj_no=t, load_com=True, difference=difference,
                                 observation_model=observation_model, order=order, max_states=max_states,
                                 dim=dim, spline_params=spline_params, prior=prior,
                                 hyperparams=None, keep_xy=keep_xy, com_savename=com_savename,
                                 radial=True, save_com=False, save_every=save_every, res=res, gro='berendsen.gro', seed_sequence=z)

        for i in traj_no:
            ihmm[i].inference(niter)
            
        for i in traj_no:
            ihmm[i]._get_params(quiet=True)

        ihmmr = [[] for i in traj_no]

        niter_fixed=10  # state sequence is fixed so parameter inference converges quick
        # convert to radial
        for i in traj_no:

            radial = np.zeros([ihmm[i].com.shape[0], 1, 2])
            radial[:, 0, 0] = np.linalg.norm(ihmm[i].com[:, 0, :2], axis=1)
            radial[:, 0, 1] = ihmm[i].com[:, 0, 2]
            
            ihmmr[i] = hdphmm.InfiniteHMM((radial, ihmm[i].dt), traj_no=[0], load_com=False, difference=False,
                                   order=order, max_states=max_states,
                                   dim=[0, 1], spline_params=spline_params, prior='MNIW-N',
                                   hyperparams=None, save_com=False, state_sequence=ihmm[i].z)

            ihmmr[i].inference(niter_fixed)

        for i in traj_no:
            ihmmr[i]._get_params(traj_no=0)

        file_rw.save_object({'ihmm': ihmm, 'ihmmr': ihmmr}, 'ihmm_%s_%diter_max_states%d_seeded_fixed.pl' % (res, niter, max_states))
        exit()

    else:

        ihmm = final_parameters['ihmm']
        ihmmr = final_parameters['ihmmr']


    # Cluster radial params

    A = None
    sigma = None
    mu = None
    T = None

    for t in range(24):

        estimated_states = ihmmr[t].z[0, :]
        found_states = list(np.unique(estimated_states))

        a = np.zeros([2, 2, len(found_states)])  # should probably an include a dimension for AR order
        s = np.zeros([2, 2, len(found_states)])
        m = np.zeros([2, len(found_states)])
        st = np.diag(ihmm[t].converged_params['T'].mean(axis=0))

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
            T = st
        else:
            A = np.concatenate((A, a), axis=-1)
            sigma = np.concatenate((sigma, s), axis=-1)
            mu = np.concatenate((mu, m), axis=-1)
            T = np.concatenate((T, st), axis=-1)

    mu_ = np.copy(mu)

    # default is diags
    eigs = False
    diags = True
    if cluster_vars == 'eigs':
        eigs = True
        diags = False

    if combine_clusters:

        params = {'sigma': sigma, 'A': A, 'mu': mu[0, :], 'T': -np.log(1 - T)}
        sig_cluster = Cluster(params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=None, nclusters=tot_clusters)        
        sig_cluster.fit()

        new_labels = sig_cluster.labels

        print('Found %d clusters' % np.unique(sig_cluster.labels).size)
        
    else:

        sig_params = {'sigma': sigma}
        A_params = {'A': A}

        sig_cluster = Cluster(sig_params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=dt_sigma, nclusters=nclusters_sigma)
        A_cluster = Cluster(A_params, eigs=eigs, diags=diags, algorithm=algorithm, distance_threshold=dt_A, nclusters=nclusters_A)
        r_cluster = Cluster({'mu':mu[0, :]}, algorithm=algorithm, nclusters=nclusters_r)
        T_cluster = Cluster({'T': -np.log(1 - T)}, algorithm=algorithm, nclusters=nclusters_T)

        sig_cluster.fit()
        A_cluster.fit()
        r_cluster.fit()
        T_cluster.fit()

        nA_clusters = np.unique(A_cluster.labels).size
        nsig_clusters = np.unique(sig_cluster.labels).size
        print('Found %d sigma clusters' % nsig_clusters)
        print('Found %d A clusters' % nA_clusters)
        print('Found %d r clusters' % nclusters_r)
        print('Found %d T clusters' % nclusters_T)

        cluster_matrix = np.zeros([nA_clusters, nsig_clusters])

        # visualize r clusters
        # print(r_cluster.labels)
        # for i in range(nclusters_r):
        #    ndx = np.where(np.array(r_cluster.labels) == i)[0]
        #    plt.hist(mu_[0, ndx])
        #plt.show()
        #exit()

        new_clusters = np.zeros([A.shape[-1]])

        for state in range(A.shape[-1]):
            #new_clusters[state] = A_cluster.labels[state] * nsig_clusters + sig_cluster.labels[state]
            new_clusters[state] = A_cluster.labels[state] * nsig_clusters * nclusters_r * nclusters_T + sig_cluster.labels[state] * nclusters_r * nclusters_T + r_cluster.labels[state] * nclusters_T + T_cluster.labels[state]
  
        print('Found %d total clusters' % np.unique(new_clusters).size)

        all_labels = np.unique(new_clusters).astype(int)

        new_label_dict = {l:i for i, l in enumerate(all_labels)}

        new_labels = [new_label_dict[int(i)] for i in new_clusters]

        sig_cluster.labels = new_labels

    all_state_params = {'A': A, 'sigma': sigma, 'mu': mu, 'state_labels': new_labels, 'T': T}

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
                                   order=order, max_states=mu.shape[0],
                                   dim=[0, 1], spline_params=spline_params, prior='MNIW',
                                   hyperparams=None, save_com=False, state_sequence=z[:, 1:])

    niter = 10  # state sequence is fixed therefore parameter inference is quick
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

    # Make sure there are no zero-rows. This can happen in the rare case where the last entry of
    # a sequence its own unique state, so it doesn't ever transition out.
    for i, row in enumerate(count_matrix):
        if row.sum() == 0:
            count_matrix[i, :] = np.ones(row.size)  # give uniform probability to transitions out of this rarely accessed state.

    # The following is very similar to ihmm3.pi_z. The difference is due to the dirichlet process.
    transition_matrix = (count_matrix.T / count_matrix.sum(axis=1)).T

    init_state = ihmm_final.z[:, 0]
    pi_init = np.zeros([nclusters])
    for i, c in enumerate(ihmm_final.found_states):
        pi_init[i] = np.where(init_state == c)[0].size

    pi_init /= pi_init.sum()

    final_parameters = {'A': A_final, 'sigma': sigma_final, 'mu': mu, 'self_T': T, 'T': transition_matrix, 'pi_init': pi_init, 'z': ihmm_final.z, 'ihmmr': ihmmr, 'ihmm':ihmm, 'all_state_params': all_state_params, 'ihmm_final': ihmm_final, 'T_distribution': ihmm_final.convergence['T']}

    if combine_clusters:

        file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_agglomerative_%s_%s_combined_%d.pl' % (res, cluster_vars, tot_clusters)) 

    else:

        if nclusters_A is None:

            file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_agglomerative_%s_%s_dtsigma%.2f_dtA%.2f.pl' %(res, cluster_vars, dt_sigma, dt_A))

        else:

            file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_agglomerative_%s_%s_nsigma%d_nA%d_nr%d_nT%d.pl' %(res, cluster_vars, nclusters_sigma, nclusters_A, nclusters_r, nclusters_T))

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
    error, _ = ts.bootstrap_msd(msd, nboot, confidence=68)

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


def cluster_behavior(params, percent, n=2):
    """
    percent: only show states that are in this percent of total trajectories
    """

    z = params['z']
    ihmmr = params['ihmmr']
    mu = params['mu']

    if res == 'MET':
        clustered_sequence = ihmmr[n].clustered_state_sequence[0, :]
        nclusters = np.unique(clustered_sequence).size
    else:
        nclusters = np.unique(z).size
        
    state_counts = dict()

    for n in range(24):

        unique_states = np.unique(ihmmr[n].clustered_state_sequence[0, :])
        #print(unique_states)
        for u in unique_states:
            if u in state_counts.keys():
                state_counts[u] += 1
            else:
                state_counts[u] = 1

    nstates = max(state_counts.keys()) + 1
    state_counts = np.array([state_counts[i] for i in range(nstates)])
    fraction = state_counts / 24
   
    prevelant_states = np.where(fraction > (percent / 100))[0]

    # For methanol
    cmap = plt.cm.jet
    if res == 'MET':
        prevelant_states = np.concatenate((prevelant_states, [14]))

        shown_colors = np.array([cmap(i) for i in np.linspace(50, 225, nclusters).astype(int)])
        colors = np.array([cmap(i) for i in np.linspace(50, 225, clustered_sequence.max() + 1).astype(int)])
        colors[np.unique(clustered_sequence)] = shown_colors
        colors[14] = colors[28] # hack

    else:
        colors = np.array([cmap(i) for i in np.linspace(50, 225, nclusters + 1).astype(int)])

    print('Prevelant States:', prevelant_states)

    #count = np.zeros(np.unique(z).size)
    
    #for i in range(z.shape[0]):
    #    for j in range(count.size):
    #        count[j] += len(np.where(z[i, :] == j)[0])

    #count /= count.sum()

    #sorted_ndx = np.argsort(count)[::-1]

    #stop = np.where(np.cumsum(count[sorted_ndx]) > (top_percent / 100))[0][0]
    #prevelant_states = sorted_ndx[:(stop + 1)]

    shift = 0.75

    fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharey=False, gridspec_kw={'width_ratios': [1, 1, 0.15]})
    #fig, ax = plt.subplots(1, 2, figsize=(7, 7), sharey=True)

    trajectory_generator = GenARData(params=final_parameters)
    
    A = final_parameters['A']
    sigma = final_parameters['sigma']
    T = final_parameters['T']
    
    fig1, Tax = plt.subplots()
    fig2, Aax = plt.subplots()
    #sigax = Aax.twinx()

    bin_width = 0.2

    #for i, p in enumerate(np.unique(z)):
    #    if mu[p, 0] > 2:
    #        print(i, p)
    #        print(0.5 * dwell(T[p, p]), mu[p, 0], state_counts[i])
    #exit()

    if res == 'MET':
        new_order = [0, 2, 4, 5, 1, 3, 6]
        prevelant_states = prevelant_states[new_order]

    for i, s in enumerate(prevelant_states):

        Adiag = np.diag(A[s, 0, ...])
        sigdiag = np.diag(sigma[s, ...])

        print(np.diag(A[s, 0, ...]), np.diag(sigma[s, ...]))
        trajectory_generator.gen_trajectory(100, 1, state_no=s, progress=False)

        t = trajectory_generator.traj[:, 0, :]
        t -= t.mean(axis=0)

        Tax.bar(i, 0.5 * dwell(T[s, s]), color=colors[s], edgecolor='black')

        Aax.scatter(sigdiag[0], Adiag[0], color=colors[s], edgecolor='black', s=100)
        Aax.scatter(sigdiag[1], Adiag[1], color=colors[s], edgecolor='black', s=100, marker='^')

        #Aax.bar(i - 1.5 * (bin_width), Adiag[0], bin_width, color=colors[s], edgecolor='black', alpha=1)
        #Aax.bar(i - 0.5 * (bin_width), Adiag[1], bin_width, color=colors[s], edgecolor='black', alpha=1)
        #sigax.bar(i + 0.5 * (bin_width), sigdiag[0], bin_width, color=colors[s], edgecolor='black', alpha=0.5)
        #sigax.bar(i + 1.5 * (bin_width), sigdiag[1], bin_width, color=colors[s], edgecolor='black', alpha=0.5)

        ax[0].plot(t[:, 1] + i*shift, lw=2, color=colors[s])
        ax[1].plot(t[:, 0] + i*shift, lw=2, color=colors[s])
        ax[2].text(0, i*shift, '%.1f %%' % (100*fraction[s]), fontsize=16, horizontalalignment='center')
  
    ax[0].set_yticks([i*shift for i in range(len(prevelant_states))])
    #ax[0].set_yticklabels(['%.1f %%' % (100*fraction[s]) for s in prevelant_states])
    
    ax[0].set_yticklabels(['%d' % (s + 1) for s in range(len(prevelant_states))])

    ax[1].set_yticks([i * shift for i in range(len(prevelant_states))])
    ax[1].set_yticklabels(['%.1f' % mu[p, 0] for p in prevelant_states])

    ax[0].set_xlabel('Step Number', fontsize=18)
    ax[0].set_ylabel('State Number', fontsize=18)
    ax[1].set_xlabel('Step Number', fontsize=18)
    ax[0].set_title('$z$ direction', fontsize=18)
    ax[1].set_title('$r$ direction', fontsize=18)
    ax[0].tick_params(labelsize=16)
    ax[1].tick_params(labelsize=16)
    ax[1].set_ylabel('Cluster radial mean', fontsize=18)
    ax[2].axis('off')
    ax[2].set_yticks([i*shift for i in range(len(prevelant_states))])
    ax[2].set_title('Percentage\nPrevalence', fontsize=18)

    ax[2].set_xlim(0, 1)

    ax[0].set_ylim(-shift, shift * len(prevelant_states))
    ax[1].set_ylim(-shift, shift * len(prevelant_states))
    ax[2].set_ylim(-shift, shift * len(prevelant_states))

    circle = mlines.Line2D([], [], color='white', markeredgecolor='black', marker='o', linestyle=None, label='radial dimension', markersize=12)
    square = mlines.Line2D([], [], color='white', markeredgecolor='black', marker='^', linestyle=None, label='axial dimension', markersize=12)

    Tax.set_ylabel('Average dwell time (ns)', fontsize=18)
    Tax.set_xticklabels([])
    
    Aax.set_ylabel('A diagonals', fontsize=18)
    Aax.set_xlabel('$\Sigma$ diagonals', fontsize=18)
    #Aax.set_xticklabels([])
    Aax.tick_params(labelsize=16)
    Tax.tick_params(labelsize=16)
    Aax.legend(handles=[circle, square], fontsize=16)
    #sigax.tick_params(labelsize=14)
    Tax.set_xticks([i for i in range(len(prevelant_states))])
    Tax.set_xticklabels([i + 1 for i in range(len(prevelant_states))])
    Tax.set_xlabel('State Number', fontsize=16)

    fig1.tight_layout()
    fig2.tight_layout()
    fig.tight_layout()

    fig.savefig('/home/ben/github/LLC_Membranes/Ben_Manuscripts/hdphmm/figures/common_states_%s.pdf' % res)
    fig1.savefig('/home/ben/github/LLC_Membranes/Ben_Manuscripts/hdphmm/figures/dwell_times_%s.pdf' % res)
    fig2.savefig('/home/ben/github/LLC_Membranes/Ben_Manuscripts/hdphmm/figures/A_sigma_scatter_%s.pdf' % res)

    plt.show()


def dwell(p, ntrials=1000): 
    """ Calculate the average length that a particle stays in the same state given a self-transition probability.
    """

    dwell_times = np.zeros(ntrials) 

    for t in range(ntrials): 

        cont = True 
        n = 0 
        while cont: 
            n += 1 
            cont = bool(np.random.choice([1, 0], p=[p, 1 -p])) 

        dwell_times[t] = n 

    return dwell_times.mean() 
    # return np.percentile(dwell_times, 95)


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

load=False
view_clusters_= False
show_states = []
calculate_msd=True
plot_realization_=True
view_cluster_behavior=False#False
top_percent = 34
recluster_= False
test_clusters_ = False #True

# This determines what file will be loaded
combine_clusters = True
tot_clusters = 30

# Reclustering parameters
combine_clusters_new = False 

try: 
    new_tot_clusters = int(sys.argv[3])
except IndexError:
    new_tot_clusters = 30 

# if these are None, the distance threshold will be used
# This will dictate what file is loaded
nclusters_A = 5 
nclusters_sigma = 5
nclusters_r = 3

# for reclustering
new_nclusters_A = 3 
new_nclusters_sigma = 5 
new_nclusters_r = 2
new_cluster = 1 

# This will determine which file is loaded
dtA = 0.25
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
            print('hi')
            final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_dtsigma%.2f_dtA%.2f.pl' % (res, cluster_vars, dtsigma, dtA))

        else:

            if nclusters_sigma is None:
                raise Exception('nclusters_sigma must not be None')

            final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_nsigma%d_nA%d_nr%d.pl' % (res, cluster_vars, nclusters_sigma, nclusters_A, nclusters_r))

else:

    final_parameters = ihmm(res, cluster=cluster, niter=niter, algorithm=algorithm, dt_A=dtA, dt_sigma=dtsigma, nclusters_sigma=nclusters_sigma, nclusters_A=nclusters_A, tot_clusters=tot_clusters,combine_clusters=combine_clusters)

if view_clusters_:

    view_clusters(final_parameters, show_states=show_states)

if view_cluster_behavior:
    cluster_behavior(final_parameters, percent=top_percent)

if recluster_:
    final_parameters = ihmm(res, cluster=new_cluster, niter=niter, algorithm=algorithm, dt_A=new_dtA, dt_sigma=new_dtsigma, final_parameters=final_parameters, nclusters_sigma=new_nclusters_sigma, nclusters_A=new_nclusters_A, tot_clusters=new_tot_clusters, combine_clusters=combine_clusters_new, nclusters_r=nclusters_r)

if test_clusters_:

    if nclusters_A is None:
        test_cluster(final_parameters, dt_A=dtA, dt_sigma=dtsigma)
    else:
        test_cluster(final_parameters, nclusters_sigma=nclusters_sigma, nclusters_A=nclusters_A)

if calculate_msd:
    msd(final_parameters, res)

if plot_realization_:

    plot_realization(final_parameters, res)
