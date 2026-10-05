"""Basic tools for working with binned neutron spectra.

Energies are in MeV. A spectrum is stored as fluence per energy bin on a set
of bin edges.
"""

import numpy as np


def log_bins(e_min, e_max, n):
    """Bin edges (MeV), evenly spaced on a log scale."""
    return np.logspace(np.log10(e_min), np.log10(e_max), n + 1)


def bin_midpoints(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def bin_widths(edges):
    return np.diff(edges)


def maxwellian_pdf(E, T):
    """Normalised Maxwellian spectrum, per MeV.

    phi(E) = 2 / (sqrt(pi) * T^1.5) * sqrt(E) * exp(-E / T)
    Integrates to 1; mean energy is 1.5 * T.
    """
    E = np.asarray(E, dtype=float)
    return 2.0 / (np.sqrt(np.pi) * T ** 1.5) * np.sqrt(E) * np.exp(-E / T)


def spectrum_from_function(edges, func, *args):
    """Fluence per bin from a spectral shape func(E, *args) given per MeV."""
    return func(bin_midpoints(edges), *args) * bin_widths(edges)


def total_fluence(phi_bins):
    return float(np.sum(phi_bins))


def mean_energy(edges, phi_bins):
    """Fluence-weighted mean energy (MeV)."""
    return float(np.sum(phi_bins * bin_midpoints(edges)) / np.sum(phi_bins))


def normalise(phi_bins):
    """Scale so the total fluence is 1."""
    return phi_bins / np.sum(phi_bins)