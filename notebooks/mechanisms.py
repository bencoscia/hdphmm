#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from LLC_Membranes.llclib import file_rw
from LLC_Membranes.timeseries.coordinate_trace import CoordinateTrace
from LLC_Membranes.llclib import timeseries as ts
import hdphmm

first_frame = {'MET': 7000, 'URE': 2000, 'GCL': 2400, 'ACH': 4000}

carb = 1
head = 2
tails = 3
monomer_colors = {'O': head, 'O1': head, 'O2': head, 'O3': carb, 'O4': carb, 'O5': tails, 'O6': tails, 'O7': tails, 'O8': tails, 'O9': tails, 'O10': tails}

res = 'MET'
nsolute = 24
res_no = np.arange(nsolute)#[0]
cluster_vars = 'diags'
distance_threshold = 0.25
fontsize=14
dt_sigma = 0.25
dt_A = 1.0
nclusters_A = 6
nclusters_sigma = 7  
combine_clusters = False
tot_clusters = 30

if combine_clusters:

    final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_combined_%d.pl' % (res, cluster_vars, tot_clusters))

else:

    if nclusters_A is None:
        
        final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_dtsigma%.2f_dtA%.2f.pl' % (res, cluster_vars, dt_sigma, dt_A))
    
    else:
        final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_nsigma%d_nA%d.pl' % (res, cluster_vars, nclusters_sigma, nclusters_A))

hbonds = file_rw.load_object('trajectories/hbond_summary_%s.pl' % res)
hbonded = hbonds['hbonded']
hbonded_to = hbonds['bonded_to']
dens = file_rw.load_object('trajectories/local_density_%s.pl' % res)
ma = 50

density = np.zeros([dens.shape[0] - ma + 1, dens.shape[1]])
for s in range(nsolute):
    density[:, s] = ts.calculate_moving_average(dens[:, s], ma)

# for color-coding types of hbonds
monomer_hbonds = np.zeros_like(hbonded, dtype=int)
for h in range(len(hbonded_to)):
    ndx = []
    for s in range(nsolute):
        if len(hbonded_to[h][s]) > 0:
            monomer_hbonds[s, h] = monomer_colors[hbonded_to[h][s][0][0]]
            ndx.append(s)

trace = file_rw.load_object('trajectories/%s_trace.pl' % res)
coord = file_rw.load_object('trajectories/coord_summary_%s.pl' % res)['coordinated']

ihmmr = final_parameters['ihmmr']

t = np.arange(ihmmr[0].com.shape[0] - 1) * ihmmr[0].dt / 1000

state_counts = dict()
for n in res_no:

    unique_states = np.unique(ihmmr[n].clustered_state_sequence[0, :])

    for u in unique_states:
        if u in state_counts.keys():
            state_counts[u] += 1
        else:
            state_counts[u] = 1

nstates = max(state_counts.keys()) + 1
state_counts = np.array([state_counts[i] for i in range(nstates)])
fraction = state_counts / 24
print(fraction[fraction > 0.5])
exit()

for n in res_no:

    zax = 1
    rax = 2
    cax = 0

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
  
    clustered_sequence = ihmmr[n].clustered_state_sequence[0, :]

    print(np.unique(clustered_sequence))
    nclusters = np.unique(clustered_sequence).size

    print('Found %d clusters' % nclusters)

    cmap = plt.cm.jet
    
    #colors = np.array([cmap(i) for i in np.random.choice(np.arange(cmap.N), size=clustered_sequence.max())])
    #colors = np.array([cmap(i) for i in np.linspace(0, cmap.N, nclusters).astype(int)])
    shown_colors = np.array([cmap(i) for i in np.linspace(50, 225, nclusters).astype(int)])
    colors = np.array([cmap(i) for i in np.linspace(50, 225, clustered_sequence.max() + 1).astype(int)])
    colors[np.unique(clustered_sequence)] = shown_colors

    for dim, a in zip([1, 0], [zax, rax]):
        com = ihmmr[n].com[1:, 0, dim]
        ax[a].add_collection(
                hdphmm.multicolored_line_collection(t, com, clustered_sequence, colors))

        ax[a].set_xlim([0, ihmmr[n].nT * ihmmr[n].dt / 1000])
        ax[a].set_ylim([com.min(), com.max()])

    ax[zax].set_ylabel('$z$ coordinate', fontsize=fontsize)
    ax[rax].set_ylabel('$r$ coordinate', fontsize=fontsize)
    ax[2].set_xlabel('Time (ns)', fontsize=fontsize)
    
    ax[zax].tick_params(labelsize=fontsize)
    ax[rax].tick_params(labelsize=fontsize)

    # hbonds 'colorbar'
    h = np.zeros(hbonded[n, :].size)
    colors = np.array([mcolors.to_rgba(i) for i in ['white', 'xkcd:blue', 'xkcd:green', 'xkcd:red']])

    #ax[cax].add_collection(hdphmm.multicolored_line_collection(t, h, hbonded[n, :].astype(int), colors, lw=20))
    ax[cax].add_collection(hdphmm.multicolored_line_collection(t, h, monomer_hbonds[n, :].astype(int), colors, lw=20))

    # sodium coordination
    colors = np.array([mcolors.to_rgba(i) for i in ['white', 'xkcd:green']])

    na = np.zeros_like(h) + 0.5
    print(coord[n, :].sum())
    ax[cax].add_collection(hdphmm.multicolored_line_collection(t, na, coord[n, :].astype(int), colors, lw=20))

    # local density 'colorbar'

    dens = np.zeros(h.size - ma) - 0.5
    #tr = ts.calculate_moving_average(density[:, n], ma)
    tr = density[:, n]

    maxi = density.mean() + 2 * density.std()
    mini = density.mean() - 2 * density.std()

    norm = plt.Normalize(mini, maxi)
    #norm = plt.Normalize(tr.min(), tr.max())
    points = np.array([t[(ma //2):-(ma // 2)], dens]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='Blues', norm=norm)
    lc.set_array(tr[1:])
    lc.set_linewidth(20)
    ax[cax].add_collection(lc)

    # radial distance 'colorbar'
    d = np.zeros_like(h) - 1
    rd = trace.radial_distance[first_frame[res]:, n]
    #cmax = np.amax(trace.radial_distance[first_frame[res]:, :])
    cmax = 3
    norm = plt.Normalize(0, cmax)
    points = np.array([t, d]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma_r', norm=norm)
    lc.set_array(rd[1:])
    lc.set_linewidth(20)
    ax[cax].add_collection(lc)
    
    ax[cax].set_xlim(0, ihmmr[n].nT * ihmmr[n].dt / 1000)
    ax[cax].set_ylim(-1.5, 1)

    #plt.tick_params(labelleft='off')
    #ax[cax].set_frame_on(False)
    #ax[cax].set_yticklabels([]) 
  
    ax[cax].axis('off')

    #plt.gca().axes.get_yaxis().set_visible(False)
    #ax[cax].spines['right'].set_visible(False)
    #ax[cax].spines['top'].set_visible(False)
    #ax[cax].spines['left'].set_visible(False)

    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.001)
    plt.show()
