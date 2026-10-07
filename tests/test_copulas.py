import numpy as np
from scipy.stats import norm, multivariate_normal as mvn


from componergy.indices.copulas import FAMILIES, compare_families


def _gaussian_copula_data(rho, n, seed):
    rng = np.random.default_rng(seed)
    xy = mvn.rvs(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=n, random_state=rng)
    return norm.cdf(xy[:, 0]), norm.cdf(xy[:, 1])


def test_copula_axioms_hold_for_all_families():
    test_params = {
        "frank": {"theta": 3.0}, "gaussian": {"rho": 0.5},
        "clayton": {"theta": 2.0, "rotated": False}, "gumbel": {"theta": 2.0, "rotated": False},
    }
    u_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    v_grid = np.array([0.2, 0.4, 0.5, 0.6, 0.8])

    for name, fam in FAMILIES.items():
        params = test_params[name]
        cdf = fam["cdf"]

        c_u_lo = cdf(u_grid, np.full_like(u_grid, 1e-4), params)
        c_hi_v = cdf(np.full_like(v_grid, 1 - 1e-4), v_grid, params)
        c_u_hi = cdf(u_grid, np.full_like(u_grid, 1 - 1e-4), params)
        assert np.allclose(c_u_lo, 0, atol=1e-2), name
        assert np.allclose(c_hi_v, v_grid, atol=1e-2), name
        assert np.allclose(c_u_hi, u_grid, atol=1e-2), name

        c_grid = cdf(u_grid, v_grid, params)
        lower_bound = np.maximum(u_grid + v_grid - 1, 0)
        upper_bound = np.minimum(u_grid, v_grid)
        assert np.all(c_grid >= lower_bound - 1e-6), name
        assert np.all(c_grid <= upper_bound + 1e-6), name


def test_gaussian_cdf_matches_independent_scipy_reference():
    rho = 0.6
    test_pts_x, test_pts_y = np.array([-1.0, 0.0, 0.5]), np.array([-0.5, 0.2, 1.0])
    test_u, test_v = norm.cdf(test_pts_x), norm.cdf(test_pts_y)

    our_cdf = FAMILIES["gaussian"]["cdf"](test_u, test_v, {"rho": rho})
    ref = mvn(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    scipy_cdf = np.array([ref.cdf([x, y]) for x, y in zip(test_pts_x, test_pts_y)])

    assert np.allclose(our_cdf, scipy_cdf, atol=1e-6)


def test_gaussian_fit_recovers_true_rho():
    u, v = _gaussian_copula_data(rho=0.6, n=2000, seed=5)
    fitted = FAMILIES["gaussian"]["fit"](u, v)
    assert abs(fitted["rho"] - 0.6) < 0.05


def test_archimedean_families_show_monotonic_dependence_recovery():
    for family_name in ["frank", "clayton", "gumbel"]:
        fam = FAMILIES[family_name]
        thetas = []
        for rho in [0.1, 0.4, 0.7, 0.9]:
            u, v = _gaussian_copula_data(rho, 3000, seed=42)
            params = fam["fit"](u, v)
            thetas.append(abs(params["theta"]))
        assert all(thetas[i] < thetas[i + 1] for i in range(len(thetas) - 1)), family_name


def test_compare_families_selects_gaussian_when_data_is_gaussian_copula():
    u, v = _gaussian_copula_data(rho=0.6, n=3000, seed=7)
    ranking = compare_families(u, v)
    assert ranking[0]["family"] == "gaussian"
    assert ranking[0]["ks_pval"] > 0.01
