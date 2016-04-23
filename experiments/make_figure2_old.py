"""
Plot the true and inferred state alignments.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import os, cPickle, copy

from hips.plotting.layout import *

from data_helper import load_synth_data
from utilities import *

def plot_score_map(ax, score):

    px = 10
    K_true, K_inf = score.shape

    # Normalize the score
    nscore = normalize(score)

    im = ax.imshow(np.kron(nscore, np.ones((px,px))), cmap='Greys',
                   vmin=0, vmax=1.0)
    ax.set_xlabel('Inferred State')
    ax.set_xticks(np.arange(px/2, K_inf*px, step=5*px))
    ax.set_xticklabels(np.arange(0, K_inf, step=5))
    ax.set_ylabel('True state')
    ax.set_yticks(np.arange(px/2,K_true*px, step=5*px))
    ax.set_yticklabels(np.arange(0,K_true, step=5))

def compute_sorted_state_maps(Z_true, K_true, Z_inf, K_inf):
    score = compute_state_map(Z_true, K_true, Z_inf, K_inf)
    mt, mi, umt, umi = align_state_sequences(Z_true, K_true, Z_inf, K_inf)

    perm = np.argsort(mt)
    mi = mi[perm]

    col_perm = np.concatenate((mi, umi))
    score = score[:, col_perm]
    return score

def plot_all_score_maps(true_hmm, true_states,
                             hdp_hmm_mcmc, hdp_hmm_mcmc_eb, hdp_hmm_mf,
                             hmm_mcmc_K20, hmm_mf_K20,
                             data,
                             figdir='figures/aas'):
    """
    Plot the true and inferred transition matrices using a variety
    of inference algorithms.

    :param hmm:
    :return:
    """
    left = 0.5
    bottom = 0.5
    height = 1.0
    width = 1.0
    titlespc = 0.4

    K_true = len(np.unique(true_states))
    # Renumber so that the first state is the most common
    true_states = renumber_state_sequence_by_usage(true_states)
    # The max overlap is the number of times state 0 is visited

    # Now plot the HDP HMM with MCMC
    hdp_hmm_mcmc_states = hdp_hmm_mcmc.states_list[0].stateseq
    hdp_hmm_mcmc_states = renumber_state_sequence(hdp_hmm_mcmc_states)
    K_hdp_hmm_mcmc = get_num_states(hdp_hmm_mcmc_states)
    hdp_hmm_mcmc_map = compute_sorted_state_maps(true_states, K_true, hdp_hmm_mcmc_states, K_hdp_hmm_mcmc)

    scale = K_hdp_hmm_mcmc / float(K_true)
    fig = create_figure((left+width*scale+0.05,
                         bottom+height+titlespc+0.05))
    ax = create_axis_at_location(fig, left, bottom,
                                 width*scale,
                                 height)
    plot_score_map(ax, hdp_hmm_mcmc_map)
    ax.set_title('HDP-HMM (HMC)')
    fig.savefig(os.path.join(figdir, 'figure_2c.pdf'))

    # HDP HMM with MCMC and EB
    hdp_hmm_mcmc_eb_states = hdp_hmm_mcmc_eb.states_list[0].stateseq
    hdp_hmm_mcmc_eb_states = renumber_state_sequence(hdp_hmm_mcmc_eb_states)
    K_hdp_hmm_mcmc_eb = get_num_states(hdp_hmm_mcmc_eb_states)
    hdp_hmm_mcmc_eb_map = compute_sorted_state_maps(true_states, K_true, hdp_hmm_mcmc_eb_states, K_hdp_hmm_mcmc_eb)

    scale = K_hdp_hmm_mcmc_eb / float(K_true)
    fig = create_figure((left+width*scale+0.05,
                         bottom+height+titlespc+0.05))
    ax = create_axis_at_location(fig, left, bottom,
                                 width*scale,
                                 height)
    plot_score_map(ax, hdp_hmm_mcmc_eb_map)
    ax.set_title('HDP-HMM (EB)')
    fig.savefig(os.path.join(figdir, 'figure_2d.pdf'))

    # Now plot the HDP HMM with MF
    hdp_hmm_mf_states = hdp_hmm_mf.heldout_viterbi(data)
    hdp_hmm_mf_states = renumber_state_sequence(hdp_hmm_mf_states)
    K_hdp_hmm_mf = get_num_states(hdp_hmm_mf_states)
    hdp_hmm_mf_map = compute_sorted_state_maps(true_states, K_true, hdp_hmm_mf_states, K_hdp_hmm_mf)

    scale = K_hdp_hmm_mf / float(K_true)
    fig = create_figure((left+width*scale+0.05,
                         bottom+height+titlespc+0.05))
    ax = create_axis_at_location(fig, left, bottom,
                                 width*scale,
                                 height)
    plot_score_map(ax, hdp_hmm_mf_map)
    ax.set_title('HDP-HMM (VB)')
    fig.savefig(os.path.join(figdir, 'figure_2e.pdf'))

    # Now plot the HMM with MCMC and correct number of states
    hmm_mcmc_states = hmm_mcmc_K20.states_list[0].stateseq
    hmm_mcmc_states = renumber_state_sequence(hmm_mcmc_states)
    K_hmm_mcmc = 20
    hmm_mcmc_map = compute_sorted_state_maps(true_states, K_true, hmm_mcmc_states, K_hmm_mcmc)

    scale = K_hmm_mcmc / float(K_true)
    fig = create_figure((left+width*scale+0.05,
                         bottom+height+titlespc+0.05))
    ax = create_axis_at_location(fig, left, bottom,
                                 width*scale,
                                 height)
    plot_score_map(ax, hmm_mcmc_map)
    ax.set_title('HMM (MCMC)')
    fig.savefig(os.path.join(figdir, 'figure_2a.pdf'))

    hmm_mf_states = hmm_mf_K20.heldout_viterbi(data)
    hmm_mf_states = renumber_state_sequence(hmm_mf_states)
    K_hmm_mf = 20
    hmm_mf_map = compute_sorted_state_maps(true_states, K_true, hmm_mf_states, K_hmm_mf)

    scale = K_hmm_mf / float(K_true)
    fig = create_figure((left+width*scale+0.05,
                         bottom+height+titlespc+0.05))
    ax = create_axis_at_location(fig, left, bottom,
                                 width*scale,
                                 height)
    plot_score_map(ax, hmm_mf_map)
    ax.set_title('HMM (VB)')
    fig.savefig(os.path.join(figdir, 'figure_2b.pdf'))


# Load the data
def load_synth_results():
    hmm, data, states, data_test, states_test = load_synth_data(1000, 80, 30, 'hdp-hmm', datadir='data')

    # Load the inference results
    print "Loading inference results"
    with open('models/synth/hdp_hmm_mcmc.pkl') as f:
        hdp_hmm_mcmc = cPickle.load(f)

    with open('models/synth/hdp_hmm_mcmc_eb.pkl') as f:
        hdp_hmm_mcmc_eb = cPickle.load(f)

    with open('models/synth/hdp_hmm_meanfield_alpha4.00_gamma8.00.pkl') as f:
        hdp_hmm_mf = cPickle.load(f)

    with open('models/synth/hmm_mcmc_K20.pkl') as f:
        hmm_mcmc_K20 = cPickle.load(f)

    with open('models/synth/hmm_mf_K20_alpha4.00.pkl') as f:
        hmm_mf_K20 = cPickle.load(f)

    return hmm, data, states, data_test, states_test, hdp_hmm_mcmc, hdp_hmm_mcmc_eb, hdp_hmm_mf, hmm_mcmc_K20, hmm_mf_K20

# Figure 2: State to state maps
# hmm, data, states, data_test, states_test, hdp_hmm_mcmc, hdp_hmm_mcmc_eb, hdp_hmm_mf, hmm_mcmc_K20, hmm_mf_K20 = load_synth_results()
# plot_all_score_maps(hmm, states,
#                      hdp_hmm_mcmc[-1],
#                      hdp_hmm_mcmc_eb[-1],
#                      hdp_hmm_mf[-1],
#                      hmm_mcmc_K20[-1],
#                      hmm_mf_K20[-1],
#                      data,
#                      figdir='figures/aas')
