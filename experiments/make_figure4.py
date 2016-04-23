"""
Plot the rats trajectory and a distribution over its locations
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams.update({'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 11})


import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import brewer2mpl
from hips.plotting.layout import *
from hips.distributions.circular_distribution import CircularDistribution

from experiment_helper import load_hipp_data

def make_figure(pos, center, radius, figdir="."):
    """
    Plot the mouses trajectory and a distribution over its locations
    """

    # Plot the locations of this state
    fig = create_figure((4,1.6))

    bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
    colors = bmap.mpl_colors

    # Plot the rats trajectory
    ax = create_axis_at_location(fig, .1, 0.1, 1.3, 1.3, transparent=True)
    plt.figtext(.05/4., 1.45/1.6, "A")
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


    ax1 = create_axis_at_location(fig, 2., 0.1, 1.3, 1.3, transparent=True, box=False, ticks=False)
    plt.figtext(1.6/4., 1.45/1.6, "B")
    ax1.set_aspect(1)
    cmap = get_cmap('YlOrRd')
    remove_plot_labels(ax1)

    # Plot the empirical location distribution
    cd = CircularDistribution(center/100., radius/100.)
    cd.fit_xy(pos[:,0]/100., pos[:,1]/100.)

    # Update the pdf to be measured in m^{-2}
    A = np.pi*(radius/100.)**2
    cd.pdf *= A

    _, cs = cd.plot(ax=ax1, cmap=cmap, plot_data=False, plot_colorbar=False)

    # Plot a subset of the locations
    inds = np.random.rand(pos.shape[0]) < 0.2
    ax1.scatter(pos[inds,0]/100., pos[inds,1]/100., s=1, marker='.', c='k')

    # Add a colorbar
    ax2 = create_axis_at_location(fig, 3.4, 0.2, 0.1, 1.1)
    cb = fig.colorbar(cs, cax=ax2, ticks=np.arange(0,2.5,step=0.5))

    cb.set_label('$p(\\ell)$ [m${}^{-2}$]',size=9)
    cb.ax.tick_params(labelsize=9)

    ax1.set_title('Empirical Location Distribution',
                 fontdict={'fontsize' : 9})


    plt.savefig(os.path.join(figdir, 'figure4.pdf'))
    plt.savefig(os.path.join(figdir, 'figure4.png'))
    plt.show()


if __name__ == "__main__":
    # Load the data
    dataset = "hipp_2dtrack_a"
    N, S_train, pos_train, S_test, pos_test, center, radius = \
        load_hipp_data(dataname=dataset)

    make_figure(pos_train, center, radius)