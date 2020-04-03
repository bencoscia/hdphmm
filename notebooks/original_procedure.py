#!/usr/bin/env python

# initialize the ihmm for each trajectory

import numpy as np
import hdphmm
import mdtraj
import matplotlib.pyplot as plt
from hdphmm.generate_timeseries import GenARData
from hdphmm.utils import timeseries as ts
from LLC_Membranes.llclib import file_rw

def ihmm(res, cluster=1, niter=100):

    cluster_variables = ['sig_diags', 'sig_A_diags', 'sig_A_eigs']
    cluster_vars = cluster_variables[cluster]
    
    # We will be applying the IHMM to each tr|ajectory independently
    ihmm = [[] for i in range(24)] 

    difference = False  # take first order difference of solute trajectories
    observation_model='AR'  # assume an autoregressive model (that's the only model implemented)
    order = 1  # autoregressive order
    max_states = 100  # More is usually better
    traj_no = np.arange(24).tolist() # np.arange(10).tolist()# [2]# None # np.arange(24)#2
    first_frame = 7000  # frame after which simulation is equilibrated
    dim = [0, 1, 2]  # dimensions of trajectory to keep
    prior = 'MNIW-N'  # MNIW-N (includes means) or MNIW (forces means to zero)
    link = False  # link trajectories and add phantom state
    keep_xy = True
    save_every = 1

    # You can define a dictionary with some spline paramters
    spline_params = {'npts_spline': 10, 'save': True, 'savename': 'trajectories/spline_%s.pl' % res}

    com_savename = 'trajectories/com_xy_radial_%s.pl' % res

    com = 'trajectories/com_xy_radial_%s.pl' % res # center of mass trajectories. If it exists, we can skip loading the MD trajectory and just load this
    gro = 'berendsen.gro'

    for t in traj_no:

        ihmm[t] = hdphmm.InfiniteHMM(com, traj_no=t, load_com=True, difference=difference,
                                 observation_model=observation_model, order=order, max_states=max_states,
                                 first_frame=first_frame, dim=dim, spline_params=spline_params, prior=prior,
                                 hyperparams=None, keep_xy=keep_xy, com_savename=com_savename, gro=gro,
                                 radial=True, save_com=True, save_every=save_every, res=res)

    for i in traj_no:
        ihmm[i].inference(niter)

    for i in traj_no:
        ihmm[i]._get_params(quiet=True)

    for i in traj_no:

        radial = np.zeros([ihmm[i].com.shape[0], 1, 2])
        radial[:, 0, 0] = np.linalg.norm(ihmm[i].com[:, 0, :2], axis=1)
        radial[:, 0, 1] = ihmm[i].com[:, 0, 2]

        ihmm[i] = hdphmm.InfiniteHMM((radial, ihmm[i].dt), traj_no=[0], load_com=False, difference=False,
                                   order=1, max_states=100,
                                   dim=[0, 1], spline_params=spline_params, prior='MNIW-N',
                                   hyperparams=None, save_com=False, state_sequence=ihmm[i].z)

        ihmm[i].inference(niter)

    for i in traj_no:
        ihmm[i]._get_params(traj_no=0)

    A = None
    sigma = None
    mu = None

    for t in traj_no:

        estimated_states = ihmm[t].z[0, :]
        found_states = list(np.unique(estimated_states))

        a = np.zeros([2, 2, len(found_states)])  # should probably an include a dimension for AR order
        s = np.zeros([2, 2, len(found_states)])
        m = np.zeros([2, len(found_states)])

        for i, state in enumerate(found_states):

            Amean = ihmm[t].converged_params['A'][:, 0, ..., i].mean(axis=0)
            sigmamean = ihmm[t].converged_params['sigma'][:, ..., i].mean(axis=0)

            # we want to cluster on unconditional mean
            mucond = ihmm[t].converged_params['mu'][..., i].mean(axis=0)  # conditional mean
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

    from hdphmm.cluster import Cluster

    # Reduce number of parameters via clustering.

    eigs = False
    diags = True
    params = {'A': A, 'sigma': sigma, 'mu':mu[0, :]} # only include radial mean

    if cluster_vars == 'sig_diags':
        params = {'sigma': sigma, 'mu':mu[0, :]} # only include radial mean

    elif cluster_vars == 'sig_A_eigs':
        eigs = True
        diags = False

    clusters = Cluster(params, eigs=eigs, diags=diags)
    clusters.fit()

    nclusters = np.unique(clusters.labels).size

    print('Found %d clusters' % nclusters)

    nclusters = np.unique(clusters.labels).size

    ndx = 0
    for i in traj_no:
        end = ndx + len(ihmm[i].found_states)
        labels = clusters.labels[ndx:end]
        ndx = end
        ihmm[i].reassign_state_sequence(clusters, labels=labels)

    all_mu = None
    for t in traj_no:

        m = ihmm[t].converged_params['mu'].mean(axis=0)
        phi = ihmm[t].converged_params['A'][:, 0, ..., :].mean(axis=0)

        # convert to unconditional mean
        for i in range(m.shape[1]):
            m[:, i] = np.linalg.inv(np.eye(2) - phi[..., i]) @ m[:, i]

        if all_mu is None:
            all_mu = m
        else:
            all_mu = np.concatenate((all_mu, m), axis=1)

    mu = np.zeros([nclusters, 2])
    for i in range(nclusters):
        ndx = np.where(np.array(clusters.labels) == i)[0]
        mu[i, :] = all_mu[:, ndx].mean(axis=1)

    mean_zero = []

    for t in traj_no:

        zeroed = ihmm[t].subtract_mean(traj_no=0, simple_mean=True)
        mean_zero.append(zeroed)

    mean_zero = np.array(mean_zero)

    z = None
    for t in traj_no:

        seq = ihmm[t].clustered_state_sequence[:, 1:]
        #seq = ihmm[t].z[:, 1:]
        if z is None:
            z = seq
        else:
            z = np.concatenate((z, seq), axis=0)

    ihmm_final = hdphmm.InfiniteHMM((np.moveaxis(mean_zero, 0, 1), ihmm[t].dt), traj_no=None, load_com=False, difference=False,
                                   order=1, max_states=100,
                                   dim=[0, 1], spline_params=spline_params, prior='MNIW',
                                   hyperparams=None, save_com=False, state_sequence=z)

    ihmm_final.inference(niter)

    nclusters = np.unique(ihmm_final.z).size

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
        A_final[c, 0, ...] = np.average(A[:, c, ...], axis=0, weights=weights[:, c])
        sigma_final[c, ...] = np.average(sigma[:, c, ...], axis=0, weights=weights[:, c])

    m = np.zeros_like(mu)
    for i in range(m.shape[0]):
        # m[i, :] = np.linalg.inv(np.eye(2) - A_final[i, 0, ...]) @ mu[i, :]
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

    # The following is very similar to ihmm_final.pi_z. The difference is due to the dirichlet process.
    transition_matrix = (count_matrix.T / count_matrix.sum(axis=1)).T
    init_state = ihmm_final.z[:, 0]
    pi_init = np.zeros([nclusters])
    # for c in range(nclusters):
    for i, c in enumerate(ihmm_final.found_states):
        pi_init[i] = np.where(init_state == c)[0].size
    
    pi_init /= pi_init.sum()

    final_parameters = {'A': A_final, 'sigma': sigma_final, 'mu': m, 'T': transition_matrix, 'pi_init': pi_init, 'z': ihmm_final.z}
    file_rw.save_object(final_parameters, 'saved_parameters/final_parameters_original_%s_%s.pl' %(res, cluster_vars))

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

load=False
calculate_msd=True
import sys
res='MET'

names = {'MET': 'methanol', 'URE': 'urea', 'GCL': 'ethylene glycol', 'ACH': 'acetic acid'}

try:
    res = sys.argv[1]
except IndexError:
    res = 'MET'

try: 
    cluster = int(sys.argv[2])
except IndexError:
    cluster=1

niter=10

print('Residue %s, cluster %d' % (res, cluster))
if load:

    cluster_variables = ['sig_diags', 'sig_A_diags', 'sig_A_eigs']
    cluster_vars = cluster_variables[cluster]

    final_parameters = file_rw.load_object('saved_parameters/final_parameters_original_%s_%s.pl' % (res, cluster_vars))

else:
    final_parameters = ihmm(res, cluster=cluster, niter=niter)

if calculate_msd:
    msd(final_parameters, res)

