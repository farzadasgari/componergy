import numpy as np

from componergy.indices.gringorten import gringorten_standardize


def test_output_is_approximately_standard_normal():
    rng = np.random.default_rng(0)
    x = rng.exponential(scale=3.0, size=200)
    z = gringorten_standardize(x)
    assert abs(z.mean()) < 0.15
    assert abs(z.std() - 1.0) < 0.1


def test_preserves_rank_order():
    rng = np.random.default_rng(0)
    x = rng.exponential(scale=3.0, size=200)
    z = gringorten_standardize(x)
    assert np.all(np.argsort(x) == np.argsort(z))


def test_nan_handling_and_minimum_sample_size():
    rng = np.random.default_rng(0)
    x = rng.exponential(scale=3.0, size=200)
    x[:5] = np.nan
    z = gringorten_standardize(x)
    assert np.isnan(z[:5]).all()
    assert not np.isnan(z[5:]).any()

    too_few = gringorten_standardize(np.array([1.0, 2.0, 3.0]))
    assert np.isnan(too_few).all()
