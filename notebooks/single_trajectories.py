#!/usr/bin/env python

import numpy as np
import hdphmm
import mdtraj
import matplotlib.pyplot as plt
from LLC_Membranes.llclib import file_rw
from hdphmm.generate_timeseries import GenARData
from hdphmm import timeseries as ts

def ihmm(res, traj_no, ntraj, hyperparams, plot=False, niter=100):

    print('Trajectory %d' % traj_no)
    difference = False  # take first order difference of solute trajectories
    observation_model='AR'  # assume an autoregressive model (that's the only model implemented)
    order = 1  # autoregressive order
    max_states = 100  # More is usually better
    dim = [0, 1, 2]  # dimensions of trajectory to keep
    prior = 'MNIW-N'  # MNIW-N (includes means) or MNIW (forces means to zero)
    link = False  # link trajectories and add phantom state
    keep_xy = True
    save_every = 1

    # You can define a dictionary with some spline paramters
    spline_params = {'npts_spline': 10, 'save': True, 'savename': 'spline_hdphmm.pl'}

    com_savename = 'trajectories/com_xy_radial_%s.pl' % res

    com = 'trajectories/com_xy_radial_%s.pl' % res  # center of mass trajectories. If it exists, we can skip loading the MD trajectory and just load this
    gro = 'berendsen.gro'

    ihmm = hdphmm.InfiniteHMM(com, traj_no=traj_no, load_com=True, difference=difference,
            observation_model=observation_model, order=order, max_states=max_states,
            dim=dim, spline_params=spline_params, prior=prior,
            hyperparams=hyperparams, keep_xy=keep_xy, com_savename=com_savename, gro=gro,
            radial=True, save_com=True, save_every=save_every)

    ihmm.inference(niter)

    #ihmm.summarize_results(traj_no=0)

    ihmm._get_params(quiet=True)
    
    radial = np.zeros([ihmm.com.shape[0], 1, 2])
    radial[:, 0, 0] = np.linalg.norm(ihmm.com[:, 0, :2], axis=1)
    radial[:, 0, 1] = ihmm.com[:, 0, 2]

    ihmmr = hdphmm.InfiniteHMM((radial, ihmm.dt), traj_no=[0], load_com=False, difference=False,
            order=1, max_states=100,
            dim=[0, 1], spline_params=spline_params, prior='MNIW-N',
            hyperparams=None, save_com=False, state_sequence=ihmm.z)

    ihmmr.inference(niter)
    #ihmmr.summarize_results(traj_no=0)
    ihmmr._get_params(traj_no=0)

    estimated_states = ihmmr.z[0, :]

    found_states = list(np.unique(estimated_states))

    # for rare cases where there is a unique state found at the end of the trajectory
    for i, f in enumerate(found_states):
    
        ndx = np.where(ihmmr.z[0, :] == f)[0]
    
        if len(ndx) == 1:
            if ndx[0] >= ihmmr.nT - 2:
                del found_states[i]
            
    ihmmr.found_states = found_states

    A = np.zeros([len(found_states), 1, 2, 2])  # should probably an include a dimension for AR order
    sigma = np.zeros([len(found_states), 2, 2])
    mu = np.zeros([len(found_states), 2])

    for i in range(len(found_states)):

        A[i, 0, ...] = ihmmr.converged_params['A'][:, 0, ..., i].mean(axis=0)
        sigma[i, ...] = ihmmr.converged_params['sigma'][:, ..., i].mean(axis=0)

        # we want to cluster on unconditional mean
        mucond = ihmmr.converged_params['mu'][..., i].mean(axis=0)  # conditional mea
        mumean = np.linalg.inv(np.eye(2) - A[i, 0, ...]) @ mucond # unconditional mean
        mu[i, :] = mumean

    nstates = len(ihmmr.found_states)

    ndx_dict = {ihmmr.found_states[i]: i for i in range(nstates)}

    count_matrix = np.zeros([nstates, nstates])

    for frame in range(1, ihmmr.nT - 1):  # start at frame 1. May need to truncate more as equilibration
        try:
            transitioned_from = [ndx_dict[i] for i in ihmmr.z[:, frame - 1]]
            transitioned_to = [ndx_dict[i] for i in ihmmr.z[:, frame]]
            for pair in zip(transitioned_from, transitioned_to):
                count_matrix[pair[0], pair[1]] += 1
        except KeyError:
            pass

    # The following is very similar to ihmm3.pi_z. The difference is due to the dirichlet process.
    transition_matrix = (count_matrix.T / count_matrix.sum(axis=1)).T

    # Initial distribution of states
    init_state = ihmmr.z[:, 0]
    pi_init = np.zeros([nstates])
    for i, c in enumerate(ihmmr.found_states):
        pi_init[i] = np.where(init_state == c)[0].size

    pi_init /= pi_init.sum()

    final_parameters = {'A': A, 'sigma': sigma, 'mu': mu, 'T': transition_matrix, 'pi_init': pi_init}

    MD_MSD = file_rw.load_object('trajectories/%s_msd.pl' % res)

    nboot = 200
    frac = 0.4
    nsteps = MD_MSD.MSD_average.shape[0]#4806
    dt = 0.5
    endshow=2000 #int(nsteps*frac)

    trajectory_generator = GenARData(params=final_parameters)
    trajectory_generator.gen_trajectory(nsteps, ntraj, bound_dimensions=[0])

    return trajectory_generator

def msd(traj_generator, nboot=200, endshow=2000, dt=0.5, show=True, show_traj=False, traj_no=0, diff=True):

    msd = ts.msd(traj, 1)
    error = ts.bootstrap_msd(msd, nboot, confidence=68)

    MD_MSD = file_rw.load_object('trajectories/%s_msd.pl' % res)

    t = np.arange(endshow)*dt

    if show_traj:
        fig, ax = plt.subplots(2, 1, figsize=(12, 5))

        trj_no = np.random.randint(100)

        ax[0].plot(traj[:, trj_no, 1], lw=2)
        ax[1].plot(traj[:, trj_no, 0], lw=2)

        ax[0].set_xlabel('Step number', fontsize=14)
        ax[0].set_ylabel('z coordinate', fontsize=14)
        ax[0].tick_params(labelsize=14)

        ax[1].set_xlabel('Step number', fontsize=14)
        ax[1].set_ylabel('r coordinate', fontsize=14)
        ax[1].tick_params(labelsize=14)
 
    if show:

        plt.plot(t, msd.mean(axis=1)[:endshow], lw=2, color='xkcd:blue')
        plt.fill_between(t, msd.mean(axis=1)[:endshow] + error[0, :endshow], msd.mean(axis=1)[:endshow] - error[1, :endshow], alpha=0.3, color='xkcd:blue')

        plt.plot(t, MD_MSD.MSD_average[:endshow], color='black', lw=2)
        plt.fill_between(t, MD_MSD.MSD_average[:endshow] + MD_MSD.limits[0, :endshow], MD_MSD.MSD_average[:endshow] - MD_MSD.limits[1, :endshow], alpha=0.3, color='black')
        if diff:
            plt.plot(t, MD_MSD.MSD[:endshow, traj_no], lw=2)
  
        plt.tick_params(labelsize=14)
        plt.xlabel('Time (ns)', fontsize=14)
        plt.ylabel('Mean Squared Displacement (nm$^2$)', fontsize=14)
        plt.title(names[res], fontsize=18)
        plt.tight_layout()

        plt.show()

    if diff:

        return np.abs(MD_MSD.MSD[endshow, traj_no] - msd.mean(axis=1)[endshow])

names = {'MET': 'methanol', 'URE': 'urea', 'GCL': 'ethylene glycol', 'ACH': 'acetic acid'}
save=False
load=True
run_ihmm=False
plot=True
combine=True #False
fit = False

import sys

try:
    res = sys.argv[1]
except IndexError:
    res = 'GCL'

print('Residue %s' % res)

niter = 100
ntraj = 100
tol = 0.05

trajectories = np.arange(24).tolist()
#trajectories = [0, 1]

if load:  # allows you to redo a single trajectory and save it with the rest

    try:
        ihmms = file_rw.load_object('trajectories/single_trajectories_%s.pl' % res)
    except FileNotFoundError:
        ihmms = [[] for _ in range(24)]
else:
    ihmms = [[] for _ in range(len(trajectories))]

if run_ihmm:

    for i, t in enumerate(trajectories):

       traj_generator = ihmm(res, t, ntraj, None, niter=niter)

       if fit:

           diff = msd(traj_generator.traj, nboot=200, endshow=2000, dt=0.5, show=True, show_traj=False, traj_no=t)
           
           while diff > tol:
               traj_generator = ihmm(res, t, ntraj, None)
               diff = msd(traj_generator.traj, nboot=200, endshow=2000, dt=0.5, show=True, show_traj=False, traj_no=t)
               print('Difference between MD_MSD and predicted MSD: %.2f' % diff)

       ihmms[i] = traj_generator

if plot:
    if combine:
        traj = ihmms[0].traj
        for t in range(1, len(trajectories)):
            traj = np.concatenate((traj, ihmms[t].traj), axis=1)

        msd(traj, diff=False) 
    else:
        for t in trajectories:
            msd(ihmms[t].traj, nboot=200, endshow=2000, dt=0.5, show=True, show_traj=False, traj_no=t)

if save:

    file_rw.save_object(ihmms, 'trajectories/single_trajectories_%s2.pl' % res)
