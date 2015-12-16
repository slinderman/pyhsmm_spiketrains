
def estimate_pos(models, S, pos, S_test, pos_test, center, radius):
    """
    Perform a linear regression from the states of the model to the position
    """
    # Use the last 50% of the samples to estimate the position
    # n_models = int(0.25*len(models))
    n_models = len(models)

    cumulative_epdf = 0
    for i,model in enumerate(models[-n_models:]):
        print "Computing predictive distribution for model: ", i
        # Compute the marginal distribution over states at each of the test
        # time steps
        expected_states_test = model.heldout_state_marginals(S_test)

        assert expected_states_test.shape[1] == model.num_states
        T,K = expected_states_test.shape

        # Compute the empirical location distributions for each of the latent states
        cds = []
        states_obj = model.states_list[0]
        state_seq = states_obj.stateseq

        for k in range(K):
            cd = CircularDistribution(center, radius)
            cd.fit_xy(pos[state_seq==k,0], pos[state_seq==k,1])
            cds.append(cd)

        epdf = np.zeros((T,) + cds[0].pdf.shape)
        for t in range(T):
            # Get the average of the location distributions at time t
            assert np.allclose(expected_states_test[t,:].sum(), 1.0)
            epdf[t,:,:] = np.array([expected_states_test[t,k]*cds[k].pdf for k in range(K)]).sum(0)

        cumulative_epdf += epdf

    # Normalize the cumulative expected location pdf
    cumulative_epdf /= n_models

    return cumulative_epdf


def compute_predictive_log_likelihood(train_data, test_data,
                                      hdp_hmm_mcmc, hdp_hmm_mcmc_eb, hdp_hmm_mf,
                                      N_samples=50):
    """

    :param true_hmm:
    :param hdp_hmm_mcmc:
    :param hdp_hmm_mcmc_eb:
    :param hdp_hmm_mf:
    :param hmm_mcmc_K20:
    :param hmm_mf_K20:
    :param test_data:
    :return:
    """
    print "Computing predictive log likelihoods"
    pll_names = []
    pll_means = []
    pll_stds = []

    # First pred ll (baseline of independent Poissons)
    from pop_distribution import ProductOfPoissons
    T,N = test_data.shape
    homog_model = ProductOfPoissons(N, alpha_0=2., beta_0=2.)
    homog_model.max_likelihood(train_data)
    baseline_ll = homog_model.log_likelihood(test_data).sum()


    # HDP HMM fit by MCMC with HMC for hyperparameters
    hdp_hmm_mcmc_plls = np.array([hmm.log_likelihood(test_data) for
                                  hmm in hdp_hmm_mcmc[-N_samples:]])
    m,s = expected_ll(hdp_hmm_mcmc_plls)
    pll_names.append('HDP-HMM (MCMC+HMC)')
    pll_means.append(m)
    pll_stds.append(s)

    # HDP HMM fit by MCMC with EB for hyperparameters
    hdp_hmm_mcmc_eb_plls = np.array([hmm.log_likelihood(test_data) for
                                  hmm in hdp_hmm_mcmc_eb[-N_samples:]])
    m,s = expected_ll(hdp_hmm_mcmc_eb_plls)
    pll_names.append('HDP-HMM (MCMC+EB)')
    pll_means.append(m)
    pll_stds.append(s)

    # HDP HMM fit by mean field
    hdp_hmm_vb = hdp_hmm_mf[-1]
    hdp_hmm_vb_smpls = []
    for n in range(N_samples):
        hdp_hmm_vb._resample_from_mf()
        hdp_hmm_vb_smpls.append(copy.deepcopy(hdp_hmm_vb))

    hdp_hmm_vb_plls = np.array([hmm.log_likelihood(test_data) for
                              hmm in hdp_hmm_vb_smpls])
    m,s = expected_ll(hdp_hmm_vb_plls)
    pll_names.append('HDP-HMM (VB)')
    pll_means.append(m)
    pll_stds.append(s)

    # Normalize by baseline pred ll to get units of bits/spk
    # improvement over the baseline of homogeneous poisson processes
    pll_means -= baseline_ll
    pll_means /= np.sum(test_data)
    pll_stds /= np.sum(test_data)

    print pll_means
    print pll_stds

    return pll_names, pll_means, pll_stds

def compute_mse_comparison():
    S_train, pos_train, S_test, pos_test, center, radius = load_rat_data()
    hdp_hmm_mcmc, hdp_hmm_mcmc_eb, hdp_hmm_mf = load_rat_results()

    cumulative_epdf = estimate_pos(hdp_hmm_mcmc[-2501:-1:500], S_train, pos_train, S_test, pos_test, center, radius)
    plot_total_position_estimates(cumulative_epdf, pos_test, center, radius, doplot=True)

    cumulative_epdf = estimate_pos(hdp_hmm_mcmc_eb[-5:], S_train, pos_train, S_test, pos_test, center, radius)
    plot_total_position_estimates(cumulative_epdf, pos_test, center, radius, doplot=False)

    # Make a list of HDP HMM samples from VB posterior
    hdp_hmm_mf_models = []
    for i in range(50):
        hdp_hmm_mf[-1]._resample_from_mf()
        hdp_hmm_mf_models.append(copy.deepcopy(hdp_hmm_mf[-1]))

    cumulative_epdf = estimate_pos(hdp_hmm_mf_models, S_train, pos_train, S_test, pos_test, center, radius)
    plot_total_position_estimates(cumulative_epdf, pos_test, center, radius, doplot=False)


def compute_pred_lls():
    S_train, pos_train, S_test, pos_test, center, radius = load_rat_data()
    hdp_hmm_mcmc, hdp_hmm_mcmc_eb, hdp_hmm_mf = load_rat_results()
    compute_predictive_log_likelihood(S_train, S_test,
                                      hdp_hmm_mcmc[-2501:-1:500],
                                      hdp_hmm_mcmc_eb[-5:],
                                      hdp_hmm_mf, N_samples=5)
