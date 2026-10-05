"""Synthetic example neutron fields built from simple analytic components."""

import numpy as np
from src.spectrum import maxwellian_pdf


def inverse_energy_pdf(E, e_lo, e_hi):
    """Normalised 1/E (epithermal) spectrum between e_lo and e_hi (MeV)."""
    E = np.asarray(E, dtype=float)
    pdf = 1.0 / (E * np.log(e_hi / e_lo))
    return np.where((E >= e_lo) & (E <= e_hi), pdf, 0.0)


def moderated_field_pdf(E, w_thermal=0.2, w_epithermal=0.3, w_fast=0.5,
                        kT=2.53e-8, e_lo=5e-7, e_hi=0.1, T_fast=1.42):
    """Illustrative moderated field: thermal Maxwellian + 1/E + fast Maxwellian.

    Weights are the fractions of total fluence in each component (sum to 1).
    kT = 0.0253 eV in MeV. This is an invented example, not a real field.
    """
    return (w_thermal * maxwellian_pdf(E, kT)
            + w_epithermal * inverse_energy_pdf(E, e_lo, e_hi)
            + w_fast * maxwellian_pdf(E, T_fast))