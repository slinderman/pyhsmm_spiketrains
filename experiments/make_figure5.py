"""
Make the first figure of the AAS paper. This is an analysis
of synthetic datasets. We show the following subfigures
1. Example of synthetic data
2. True transition matrix
3. Inferred transition matrix with HMM (MCMC)
4. Inferred transition matrix with iHMM (VB)
5. Inferred transition matrix with iHMM (MCMC + HMC)
6. Inferred transition matrix with iHMM (MCMC + EB)
"""
import numpy as np
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})


import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import os, cPickle, copy
import brewer2mpl

from hips.plotting.layout import *
from hips.plotting.colormaps import white_to_color_cmap

from hips.distributions.circular_distribution import CircularDistribution

def plot_total_position_estimates(cumulative_epdf, pos_test, center, radius,
                                  figdir='figures/aas',
                                  doplot=False):
    """
    Plot a few sequences of test locations and the model's predictions
    """
    bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
    colors = bmap.mpl_colors

    # Compute the mean trajectory
    T,_ = pos_test.shape
    t_test = np.arange(T) * 0.25
    mean_location = np.zeros_like(pos_test)
    for t in range(T):
        cd = CircularDistribution(center, radius, pdf=cumulative_epdf[t,:])
        mean_location[t,:] = cd.mean

def plot_rat_trajectory(pos, center, radius,
                       figdir='figures/aas',
                       figsize=(1.35, 1.6) ):

    bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
    colors = bmap.mpl_colors

    fig = plt.figure(figsize=figsize)
    ax = create_axis_at_location(fig, .05, 0.1, 1.25, 1.25)
    remove_plot_labels(ax)

    import matplotlib.patches
    circle = matplotlib.patches.Circle(xy=[0,0],
                                       radius= radius,
                                       linewidth=1,
                                       edgecolor="k",
                                       facecolor="none")
    ax.add_patch(circle)

    # Plot the path in blue
    T = -1
    ax.plot(pos[:T,0]-center[0], pos[:T,1]-center[1], '-k', lw=0.5, color=colors[1])

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)

    ax.set_title('Location Trajectory',
                 fontdict={'size' : 9})

    fig.savefig(os.path.join(figdir, 'figure_4_trajectory.pdf'))
    plt.close(fig)



# S_train, pos_train, S_test, pos_test, center, radius = load_rat_data()
# plot_rat_trajectory(pos_train, center, radius)

