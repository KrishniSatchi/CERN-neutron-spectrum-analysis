import numpy as np
from src.spectrum import (
    log_bins, bin_widths, maxwellian_pdf, spectrum_from_function,
    total_fluence, mean_energy, normalise,
)

T = 1.42  # MeV, illustrative
EDGES = log_bins(1e-3, 30.0, 500)


def test_bin_widths_are_positive_and_span_the_range():
    w = bin_widths(EDGES)
    assert np.all(w > 0)
    assert np.isclose(w.sum(), 30.0 - 1e-3)


def test_maxwellian_is_normalised():
    phi = spectrum_from_function(EDGES, maxwellian_pdf, T)
    assert np.isclose(total_fluence(phi), 1.0, rtol=5e-3)


def test_mean_energy_matches_analytic_value():
    phi = spectrum_from_function(EDGES, maxwellian_pdf, T)
    assert np.isclose(mean_energy(EDGES, phi), 1.5 * T, rtol=5e-3)


def test_hotter_spectrum_has_higher_mean_energy():
    cold = spectrum_from_function(EDGES, maxwellian_pdf, 1.0)
    hot = spectrum_from_function(EDGES, maxwellian_pdf, 2.0)
    assert mean_energy(EDGES, hot) > mean_energy(EDGES, cold)


def test_spectrum_is_never_negative():
    phi = spectrum_from_function(EDGES, maxwellian_pdf, T)
    assert np.all(phi >= 0)


def test_normalise_gives_unit_total():
    phi = spectrum_from_function(EDGES, maxwellian_pdf, T) * 7.3
    assert np.isclose(total_fluence(normalise(phi)), 1.0)