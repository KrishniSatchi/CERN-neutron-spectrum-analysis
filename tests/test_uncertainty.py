import numpy as np
from src.spectrum import log_bins, bin_midpoints, spectrum_from_function, normalise
from src.example_fields import moderated_field_pdf
from src.dose import interpolate_coefficients
from src.uncertainty import propagate, rebin

EDGES = log_bins(1e-9, 20.0, 200)
PHI = normalise(spectrum_from_function(EDGES, moderated_field_pdf))
TAB_E = np.array([1e-10, 1e3])
TAB_H = np.array([10.0, 500.0])
H = interpolate_coefficients(bin_midpoints(EDGES), TAB_E, TAB_H)
NOMINAL_DOSE = float(np.sum(PHI * H))


def run(rel_unc, rel_corr, n=20000, seed=1):
    return propagate(EDGES, PHI, rel_unc, rel_corr, TAB_E, TAB_H, n=n, seed=seed)


def test_zero_uncertainty_reproduces_nominal():
    r = run(0.0, 0.0, n=50)
    assert np.allclose(r["dose_pSv"], NOMINAL_DOSE)
    assert np.allclose(r["mean_energy_MeV"], r["mean_energy_MeV"][0])


def test_mean_of_samples_is_close_to_nominal():
    r = run(0.10, 0.0)
    assert np.isclose(r["dose_pSv"].mean(), NOMINAL_DOSE, rtol=0.01)


def test_independent_uncertainty_matches_analytic_std():
    b = 0.10
    r = run(b, 0.0)
    analytic = b * np.sqrt(np.sum((PHI * H) ** 2))
    assert np.isclose(r["dose_pSv"].std(ddof=1), analytic, rtol=0.03)


def test_fully_correlated_uncertainty_scales_dose_but_not_mean_energy():
    a = 0.05
    r = run(0.0, a)
    assert np.isclose(r["dose_pSv"].std(ddof=1) / r["dose_pSv"].mean(), a, rtol=0.03)
    assert r["mean_energy_MeV"].std(ddof=1) < 1e-9


def test_same_seed_gives_same_samples():
    a = run(0.1, 0.05, n=100, seed=7)["dose_pSv"]
    b = run(0.1, 0.05, n=100, seed=7)["dose_pSv"]
    assert np.array_equal(a, b)


OLD = log_bins(1e-3, 10.0, 400)
RNG_PHI = np.random.default_rng(0).random(400)


def test_rebin_conserves_total_fluence():
    new = log_bins(1e-3, 10.0, 37)
    assert np.isclose(rebin(RNG_PHI, OLD, new).sum(), RNG_PHI.sum())


def test_rebin_onto_same_edges_changes_nothing():
    assert np.allclose(rebin(RNG_PHI, OLD, OLD), RNG_PHI)


def test_rebin_merging_groups_of_bins_gives_group_sums():
    new = OLD[::10]
    expected = RNG_PHI.reshape(-1, 10).sum(axis=1)
    assert np.allclose(rebin(RNG_PHI, OLD, new), expected)


def test_rebin_onto_sub_range_keeps_only_that_fluence():
    new = OLD[200:]
    assert np.allclose(rebin(RNG_PHI, OLD, new), RNG_PHI[200:])
    