"""Counting statistics and detector validation tools (synthetic examples).

A detector counts events at random, so the number of counts in a fixed time
follows a Poisson distribution: mean N, standard deviation sqrt(N).
"""

import numpy as np
from scipy.optimize import curve_fit


def simulate_measurement(source_rate, bkg_rate, t_gross, t_bkg, size=None, seed=None):
    """Simulate Poisson counts: a gross run (source + background) and a
    background-only run. Rates in counts/s, times in s.

    source_rate can be a single number or an array (e.g. one per distance).
    Returns (gross_counts, background_counts).
    """
    rng = np.random.default_rng(seed)
    gross = rng.poisson((np.asarray(source_rate) + bkg_rate) * t_gross, size=size)
    bkg = rng.poisson(bkg_rate * t_bkg, size=size)
    return gross, bkg


def net_rate(gross_counts, t_gross, bkg_counts, t_bkg):
    """Background-subtracted rate and its 1-sigma uncertainty (counts/s).

    Uncertainty from Poisson statistics of both runs. A count of zero is
    treated as one when estimating the uncertainty.
    """
    gross = np.asarray(gross_counts, dtype=float)
    bkg = np.asarray(bkg_counts, dtype=float)
    net = gross / t_gross - bkg / t_bkg
    sigma = np.sqrt(np.maximum(gross, 1.0) / t_gross ** 2 + np.maximum(bkg, 1.0) / t_bkg ** 2)
    return net, sigma


def expected_relative_uncertainty(source_rate, bkg_rate, t_gross, t_bkg):
    """Expected relative uncertainty of the net rate (Poisson statistics)."""
    var = (source_rate + bkg_rate) / t_gross + bkg_rate / t_bkg
    return np.sqrt(var) / source_rate


def time_for_relative_uncertainty(source_rate, bkg_rate, target, bkg_time_factor=1.0):
    """Gross counting time (s) needed to reach a target relative uncertainty.

    The background run lasts bkg_time_factor times as long as the gross run.
    """
    return ((source_rate + bkg_rate) + bkg_rate / bkg_time_factor) / (target * source_rate) ** 2


def distance_response(d, A, d0, B):
    """Net count rate vs distance d (m): A / (d + d0)^2 + B.

    A:  strength (counts/s m^2).  d0: offset of the effective detector centre (m).
    B:  roughly constant extra rate, e.g. room-scattered neutrons.
    """
    return A / (np.asarray(d) + d0) ** 2 + B


def fit_distance_response(d, rate, sigma, p0=(500.0, 0.0, 1.0)):
    """Weighted fit of distance_response. Returns (params, covariance, chi2, ndf)."""
    d, rate, sigma = np.asarray(d), np.asarray(rate), np.asarray(sigma)
    popt, pcov = curve_fit(
        distance_response, d, rate, p0=p0, sigma=sigma, absolute_sigma=True,
        bounds=([0.0, -0.4, 0.0], [np.inf, 0.4, np.inf]), maxfev=20000,
    )
    chi2 = float(np.sum(((rate - distance_response(d, *popt)) / sigma) ** 2))
    return popt, pcov, chi2, len(d) - len(popt)


def validation_ratio(reading, reading_sigma, reference, reference_sigma):
    """Compare a detector reading with a reference value.

    Returns (ratio, ratio_sigma, z, agrees) where z = (ratio - 1) / ratio_sigma
    and agrees means |z| < 2.
    """
    ratio = reading / reference
    ratio_sigma = ratio * np.sqrt((reading_sigma / reading) ** 2 + (reference_sigma / reference) ** 2)
    z = (ratio - 1.0) / ratio_sigma
    return ratio, ratio_sigma, z, bool(abs(z) < 2.0)