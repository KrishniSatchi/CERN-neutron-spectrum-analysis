import numpy as np
from src.counting import (
    simulate_measurement, net_rate, expected_relative_uncertainty,
    time_for_relative_uncertainty, distance_response, fit_distance_response,
    validation_ratio,
)


def test_net_rate_known_values():
    net, sigma = net_rate(1200, 60.0, 300, 60.0)
    assert np.isclose(net, 20.0 - 5.0)
    assert np.isclose(sigma, np.sqrt(1200 / 3600 + 300 / 3600))


def test_simulated_net_rate_is_unbiased():
    gross, bkg = simulate_measurement(20.0, 5.0, 100.0, 100.0, size=5000, seed=1)
    net, _ = net_rate(gross, 100.0, bkg, 100.0)
    assert np.isclose(net.mean(), 20.0, rtol=0.01)


def test_simulated_scatter_matches_expected_uncertainty():
    gross, bkg = simulate_measurement(20.0, 5.0, 100.0, 100.0, size=5000, seed=2)
    net, _ = net_rate(gross, 100.0, bkg, 100.0)
    rel = net.std(ddof=1) / 20.0
    assert np.isclose(rel, expected_relative_uncertainty(20.0, 5.0, 100.0, 100.0), rtol=0.05)


def test_uncertainty_falls_as_one_over_root_time():
    short = expected_relative_uncertainty(20.0, 5.0, 100.0, 100.0)
    long_ = expected_relative_uncertainty(20.0, 5.0, 400.0, 400.0)
    assert np.isclose(long_, short / 2.0)


def test_time_for_target_round_trip():
    t = time_for_relative_uncertainty(20.0, 5.0, 0.05, bkg_time_factor=2.0)
    assert np.isclose(expected_relative_uncertainty(20.0, 5.0, t, 2.0 * t), 0.05)


def test_more_background_needs_longer_counting():
    low = time_for_relative_uncertainty(20.0, 1.0, 0.05)
    high = time_for_relative_uncertainty(20.0, 50.0, 0.05)
    assert high > low


def test_same_seed_gives_same_counts():
    a = simulate_measurement(20.0, 5.0, 100.0, 100.0, size=10, seed=5)
    b = simulate_measurement(20.0, 5.0, 100.0, 100.0, size=10, seed=5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_distance_fit_recovers_noise_free_parameters():
    d = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    truth = (500.0, 0.03, 2.0)
    rate = distance_response(d, *truth)
    popt, _, chi2, ndf = fit_distance_response(d, rate, np.ones_like(d))
    assert np.allclose(popt, truth, rtol=1e-3)
    assert ndf == len(d) - 3


def test_validation_ratio_agreement_and_disagreement():
    _, _, z, agrees = validation_ratio(100.0, 1.0, 100.0, 3.0)
    assert z == 0.0 and agrees
    _, _, z, agrees = validation_ratio(120.0, 1.0, 100.0, 3.0)
    assert z > 2 and not agrees