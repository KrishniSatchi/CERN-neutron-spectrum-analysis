import numpy as np
import pytest
from src.spectrum import log_bins, spectrum_from_function, normalise
from src.example_fields import moderated_field_pdf
from src.dose import (
    interpolate_coefficients, total_dose, dose_by_band,
    spectrum_averaged_coefficient,
)

EDGES = log_bins(1e-9, 20.0, 600)
RAW = spectrum_from_function(EDGES, moderated_field_pdf)
PHI = normalise(RAW)

CONST_E = np.array([1e-10, 100.0])
CONST_H = np.array([100.0, 100.0])


def test_example_field_is_roughly_normalised():
    assert np.isclose(RAW.sum(), 1.0, rtol=0.02)


def test_constant_coefficient_gives_dose_proportional_to_fluence():
    assert np.isclose(total_dose(EDGES, PHI, CONST_E, CONST_H), 100.0 * PHI.sum())


def test_spectrum_average_of_constant_coefficient_is_that_constant():
    assert np.isclose(spectrum_averaged_coefficient(EDGES, PHI, CONST_E, CONST_H), 100.0)


def test_power_law_interpolation_is_exact():
    energies = np.array([1e-3, 1e2])
    coeffs = 100.0 * np.sqrt(energies)
    assert np.isclose(interpolate_coefficients(1.0, energies, coeffs), 100.0)


def test_bands_add_up_to_total():
    bands = dose_by_band(EDGES, PHI, CONST_E, CONST_H)
    parts = sum(v for k, v in bands.items() if k != "total")
    assert np.isclose(parts, bands["total"])


def test_energy_outside_table_raises():
    with pytest.raises(ValueError):
        interpolate_coefficients(1e5, CONST_E, CONST_H)