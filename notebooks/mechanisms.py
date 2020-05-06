#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from LLC_Membranes.llclib import file_rw
from LLC_Membranes.timeseries.coordinate_trace import CoordinateTrace
from LLC_Membranes.llclib import timeseries as ts
import hdphmm
import sys

first_frame = {'MET': 7000, 'URE': 2000, 'GCL': 2400, 'ACH': 4000}

carb = 1
head = 2
tails = 3
other_solutes = 0
monomer_colors = {'O': head, 'O1': head, 'O2': head, 'O3': carb, 'O4': carb, 'O5': tails, 'O6': tails, 'O7': tails, 'O8': tails, 'O9': tails, 'O10': tails, 'N1': other_solutes, 'N': other_solutes}
nclusters_res = {'URE': {'nclustersA': 5, 'nclusters_sigma': 5, 'nclusters_r': 3}, 'MET': {'nclustersA': 5, 'nclusters_sigma': 5, 'nclusters_r': 3 }, 'ACH': {'nclustersA': 5, 'nclusters_sigma': 5, 'nclusters_r': 3}, 'GCL': {'nclustersA': 5, 'nclusters_sigma': 5, 'nclusters_r': 3}}

try:
    res = sys.argv[1]
except IndexError:
    res = 'MET'

nsolute = 24
res_no = np.arange(nsolute)#[0]
cluster_vars = 'diags'
fontsize=14
only_traj = True

dt_sigma = 0.25
dt_A = 1.0

nclusters_A = nclusters_res[res]['nclustersA'] 
nclusters_sigma = nclusters_res[res]['nclusters_sigma']
nclusters_r = nclusters_res[res]['nclusters_r']

combine_clusters = False 
tot_clusters = 15 

if combine_clusters:

    final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_combined_%d.pl' % (res, cluster_vars, tot_clusters))

else:

    if nclusters_A is None:
        
        final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_dtsigma%.2f_dtA%.2f.pl' % (res, cluster_vars, dt_sigma, dt_A))
    
    else:
        print(nclusters_A, nclusters_sigma, nclusters_r)
        final_parameters = file_rw.load_object('saved_parameters/final_parameters_agglomerative_%s_%s_nsigma%d_nA%d_nr%d.pl' % (res, cluster_vars, nclusters_sigma, nclusters_A, nclusters_r))

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

distinct_states_precluster = 0
state_counts = dict()
for n in res_no:
    
    unique_states = np.unique(ihmmr[n].clustered_state_sequence[0, :])
    distinct_states_precluster += len(np.unique(ihmmr[n].z[0, :]))
    for u in unique_states:
        if u in state_counts.keys():
            state_counts[u] += 1
        else:
            state_counts[u] = 1
print('%s distinct states in unclustered trajectories' % distinct_states_precluster)
nstates = max(state_counts.keys()) + 1
state_counts = np.array([state_counts[i] for i in range(nstates)])
#print(state_counts.size)
#exit()
fraction = state_counts / 24
print('Fraction of trajectories in which states appear:')
print(fraction)
print(np.where(fraction > 0.34)[0])

#img=mpimg.imread('monomer_oxygens.png')

#plt.imshow(img)
#plt.show()
#exit()

#print(fraction[fraction > 0.5])
#exit()
z = final_parameters['z']
unique_clusters = np.unique(z)
for n in res_no:

    zax = 1
    rax = 2
    cax = 0

#    if only_traj:
#
#        fig, ax = plt.subplots(3, 1, figsize=(12, 5), sharex=True)

#    else:

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios':[0.5, 1, 1]})
  
    clustered_sequence = ihmmr[n].clustered_state_sequence[0, :]
    #clustered_sequence = ihmmr[n].z[0, :]

    print(np.unique(clustered_sequence))

    nclusters = np.unique(clustered_sequence).size
    #nclusters = unique_clusters.size
    print('Found %d clusters' % nclusters)
    print('Originall there were %d clusters' % len(np.unique(ihmmr[n].z)))

    cmap = plt.cm.jet
    
    #colors = np.array([cmap(i) for i in np.random.choice(np.arange(cmap.N), size=clustered_sequence.max())])
    #colors = np.array([cmap(i) for i in np.linspace(0, cmap.N, nclusters).astype(int)])
    shown_colors = np.array([cmap(i) for i in np.linspace(50, 225, nclusters).astype(int)])
    colors = np.array([cmap(i) for i in np.linspace(50, 225, clustered_sequence.max() + 1).astype(int)])
    #colors = np.array([cmap(i) for i in np.linspace(50, 225, unique_clusters.max() + 1).astype(int)])
    colors[np.unique(clustered_sequence)] = shown_colors
    #colors[unique_clusters] = shown_colors

    for dim, a in zip([1, 0], [zax, rax]):
        com = ihmmr[n].com[1:, 0, dim]
        #ax[a].add_collection(
        #        hdphmm.multicolored_line_collection(t, com, clustered_sequence, colors))

        ax[a].plot(t, com, lw=2, color='xkcd:blue')

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

    na = np.zeros_like(h) + 0.25
    print(coord[n, :].sum())
    ax[cax].add_collection(hdphmm.multicolored_line_collection(t, na, coord[n, :].astype(int), colors, lw=20))

    ax[cax].set_yticks([-0.25, 0, 0.25])
    #ax[cax].set_yticklabels(['   $r$      ', r'     $\rho$     ', 'H-bonds', '    Na$^+$    \nAssociation'])
    ax[cax].set_yticklabels([r'     $\rho$     ', 'H-bonds', '    Na$^+$    \nAssociation'])
    ax[cax].tick_params(left=False, labelsize=14, bottom=False)
    #ax[cax].text(t[t.size // 2], 0.5, 'TEST!', fontsize=14)

    # local density 'colorbar'

    dens = np.zeros(h.size - ma) - 0.25
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
    #d = np.zeros_like(h) - 1
    #rd = trace.radial_distance[first_frame[res]:, n]
    #cmax = np.amax(trace.radial_distance[first_frame[res]:, :])
    #cmax = 3
    #norm = plt.Normalize(0, cmax)
    #points = np.array([t, d]).T.reshape(-1, 1, 2)

    #segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #lc = LineCollection(segments, cmap='plasma_r', norm=norm)
    #lc.set_array(rd[1:])
    #lc.set_linewidth(20)
    #ax[cax].add_collection(lc)
    
    ax[cax].set_xlim(0, ihmmr[n].nT * ihmmr[n].dt / 1000)
    ax[cax].set_ylim(-0.35,.35)

    #plt.tick_params(labelleft='off')
    ax[cax].set_frame_on(False)
    #ax[cax].set_yticklabels([]) 
  
    #ax[cax].axis('off')

    #plt.gca().axes.get_yaxis().set_visible(False)
    #ax[cax].spines['right'].set_visible(False)
    #ax[cax].spines['top'].set_visible(False)
    #ax[cax].spines['left'].set_visible(False)
    #ax[cax].spines['bottom'].set_visible(False)

    plt.tight_layout()

    #if n == 2:
    #    plt.savefig('/home/ben/github/LLC_Membranes/Ben_Manuscripts/hdphmm/mechanism_map.png')
    #    exit()
    #plt.subplots_adjust(hspace=0.001)
    plt.show()
