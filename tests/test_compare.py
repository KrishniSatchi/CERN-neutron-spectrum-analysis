import numpy as np
from src.spectrum import log_bins, spectrum_from_function, normalise, maxwellian_pdf
from src.example_fields import inverse_energy_pdf
from src.uncertainty import sample_spectra
from src.compare import compare_spectra, group_ratios, group_z_scores, dose_ratio_distribution

EDGES = log_bins(1e-3, 20.0, 400)
GROUPS = log_bins(1e-3, 20.0, 8)
TAB_E = np.array([1e-4, 1e3])
TAB_H = np.array([10.0, 500.0])
REF = normalise(spectrum_from_function(EDGES, maxwellian_pdf, 1.42))


def test_identical_spectra_compare_as_equal():
    c = compare_spectra(EDGES, REF, REF, TAB_E, TAB_H)
    assert np.isclose(c["fluence_ratio"], 1.0)
    assert np.isclose(c["dose_ratio"], 1.0)
    assert np.allclose(group_ratios(EDGES, REF, REF, GROUPS), 1.0)


def test_scaled_spectrum_changes_normalisation_but_not_shape():
    c = compare_spectra(EDGES, REF, 1.1 * REF, TAB_E, TAB_H)
    assert np.isclose(c["fluence_ratio"], 1.1)
    assert np.isclose(c["dose_ratio"], 1.1)
    assert np.isclose(c["avg_coeff_ratio"], 1.0)
    assert np.isclose(c["mean_energy_test_MeV"], c["mean_energy_ref_MeV"])
    assert np.allclose(group_ratios(EDGES, REF, 1.1 * REF, GROUPS), 1.0)


def test_low_energy_scatter_lowers_mean_energy_and_tilts_shape():
    def pdf(E):
        return 0.85 * maxwellian_pdf(E, 1.42) + 0.15 * inverse_energy_pdf(E, 1e-3, 0.5)
    test = normalise(spectrum_from_function(EDGES, pdf))
    c = compare_spectra(EDGES, REF, test, TAB_E, TAB_H)
    assert c["mean_energy_test_MeV"] < c["mean_energy_ref_MeV"]
    r = group_ratios(EDGES, REF, test, GROUPS)
    assert r[0] > 1.0 and r[-1] < 1.0


def test_unbiased_samples_give_small_z_scores():
    s = sample_spectra(REF, 0.10, 0.0, n=5000, seed=3)
    z = group_z_scores(EDGES, REF, s, GROUPS)
    assert np.all(np.abs(z) < 0.1)


def test_biased_samples_give_large_z_scores():
    s = sample_spectra(1.3 * REF, 0.10, 0.0, n=5000, seed=3)
    z = group_z_scores(EDGES, REF, s, GROUPS)
    assert np.all(z > 2.5)


def test_dose_ratio_distribution_centres_on_scale_factor():
    s = sample_spectra(1.2 * REF, 0.05, 0.0, n=2000, seed=4)
    r = dose_ratio_distribution(EDGES, REF, s, TAB_E, TAB_H)
    assert np.isclose(r.mean(), 1.2, rtol=0.01)