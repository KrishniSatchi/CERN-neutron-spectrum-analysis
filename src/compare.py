"""Tools for comparing a test spectrum against a reference spectrum.

All spectra are fluence per bin on the same bin edges.
"""

import numpy as np
from src.spectrum import bin_midpoints, mean_energy
from src.dose import dose_by_band, interpolate_coefficients
from src.uncertainty import rebin


def compare_spectra(edges, phi_ref, phi_test, energies, coeffs):
    """Headline numbers comparing a test spectrum with a reference.

    Ratios are test / reference.
    """
    d_ref = dose_by_band(edges, phi_ref, energies, coeffs)
    d_test = dose_by_band(edges, phi_test, energies, coeffs)
    out = {
        "fluence_ratio": float(np.sum(phi_test) / np.sum(phi_ref)),
        "mean_energy_ref_MeV": mean_energy(edges, phi_ref),
        "mean_energy_test_MeV": mean_energy(edges, phi_test),
        "dose_ratio": d_test["total"] / d_ref["total"],
        "avg_coeff_ref_pSv_cm2": d_ref["total"] / float(np.sum(phi_ref)),
        "avg_coeff_test_pSv_cm2": d_test["total"] / float(np.sum(phi_test)),
    }
    out["avg_coeff_ratio"] = out["avg_coeff_test_pSv_cm2"] / out["avg_coeff_ref_pSv_cm2"]
    for band in ("thermal", "epithermal", "fast", "high"):
        if d_ref[band] > 0:
            out[f"dose_ratio_{band}"] = d_test[band] / d_ref[band]
    return out


def group_ratios(edges, phi_ref, phi_test, group_edges, shape_only=True):
    """Test / reference fluence in coarse energy groups.

    shape_only=True scales both spectra to unit total fluence first, so the
    ratio shows differences in SHAPE only, not overall normalisation.
    """
    ref = rebin(phi_ref, edges, group_edges)
    test = rebin(phi_test, edges, group_edges)
    if shape_only:
        ref, test = ref / ref.sum(), test / test.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ref > 0, test / ref, np.nan)


def group_z_scores(edges, phi_ref, samples_test, group_edges):
    """Per group: (mean of test samples - reference) / spread of test samples.

    |z| well above 2 means the difference is larger than the stated
    uncertainty of the test spectrum. samples_test comes from
    src.uncertainty.sample_spectra. Groups are treated one at a time; the
    correlations between groups are not accounted for.
    """
    ref = rebin(phi_ref, edges, group_edges)
    s = rebin(samples_test, edges, group_edges)
    std = s.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, (s.mean(axis=0) - ref) / std, np.nan)


def dose_ratio_distribution(edges, phi_ref, samples_test, energies, coeffs):
    """Dose(test sample) / Dose(reference), one value per sample."""
    h = interpolate_coefficients(bin_midpoints(edges), energies, coeffs)
    return (samples_test * h).sum(axis=1) / float(np.sum(phi_ref * h))