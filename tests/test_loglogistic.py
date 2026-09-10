import numpy as np
from scipy.stats import fisk, kstest

from componergy.indices.loglogistic import fit_loglogistic_lmoments, loglogistic_cdf


def test_recovers_true_parameters_from_known_distribution():
    # scipy's Fisk distribution IS the log-logistic distribution:
    # F(x) = 1 / (1 + ((x-loc)/scale)^(-c)), matching loglogistic_cdf
    # with alpha=scale, beta=c, gamma=loc.
    true_beta, true_gamma, true_alpha = 4.5, -50.0, 30.0
    rng = np.random.default_rng(0)
    samples = fisk.rvs(c=true_beta, loc=true_gamma, scale=true_alpha, size=20000, random_state=rng)

    alpha, beta, gamma_param = fit_loglogistic_lmoments(samples)

    assert abs(alpha - true_alpha) / true_alpha < 0.1
    assert abs(beta - true_beta) / true_beta < 0.1
    assert abs(gamma_param - true_gamma) < 5


def test_fitted_cdf_is_approximately_uniform_on_its_own_data():
    true_beta, true_gamma, true_alpha = 4.5, -50.0, 30.0
    rng = np.random.default_rng(0)
    samples = fisk.rvs(c=true_beta, loc=true_gamma, scale=true_alpha, size=20000, random_state=rng)

    alpha, beta, gamma_param = fit_loglogistic_lmoments(samples)
    u = loglogistic_cdf(samples, alpha, beta, gamma_param)

    _, ks_pval = kstest(u, "uniform")
    assert ks_pval > 0.01


def test_raises_on_too_few_samples():
    import pytest
    with pytest.raises(ValueError):
        fit_loglogistic_lmoments(np.array([1.0, 2.0, 3.0]))
