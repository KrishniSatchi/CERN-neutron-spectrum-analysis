"""Monte Carlo uncertainty propagation and fluence-conserving rebinning for
binned neutron spectra.

Monte Carlo idea: build many randomly perturbed copies of the spectrum, push
each one through the calculation (dose, mean energy, ...), and look at the
spread of the answers.
"""

import numpy as np
import pandas as pd
from src.spectrum import bin_midpoints
from src.dose import interpolate_coefficients, BANDS


def sample_spectra(phi, rel_unc, rel_corr=0.0, n=10000, seed=42):
    """Draw n randomly perturbed copies of a binned spectrum.

    phi:      nominal fluence per bin.
    rel_unc:  relative 1-sigma uncertainty of each bin, independent from bin
              to bin (a single number, or one number per bin).
    rel_corr: relative 1-sigma uncertainty shared by ALL bins at once, e.g. a
              normalisation or calibration uncertainty.
    Negative values (rare, only for very large uncertainties) are set to zero.
    Returns an array of shape (n, number_of_bins).
    """
    phi = np.asarray(phi, dtype=float)
    rng = np.random.default_rng(seed)
    common = 1.0 + rel_corr * rng.standard_normal((n, 1))
    indep = 1.0 + np.asarray(rel_unc) * rng.standard_normal((n, phi.size))
    return np.clip(phi * common * indep, 0.0, None)


def propagate(edges, phi, rel_unc, rel_corr, energies, coeffs, n=10000, seed=42):
    """Push perturbed spectra through the calculations.

    Returns a dict of arrays (one value per random sample):
    total fluence, mean energy, total dose, spectrum-averaged coefficient,
    and the dose in each energy band that contains bins.
    """
    samples = sample_spectra(phi, rel_unc, rel_corr, n, seed)
    mid = bin_midpoints(edges)
    h = interpolate_coefficients(mid, energies, coeffs)
    dose_bins = samples * h
    total_phi = samples.sum(axis=1)

    out = {
        "total_fluence": total_phi,
        "mean_energy_MeV": (samples * mid).sum(axis=1) / total_phi,
        "dose_pSv": dose_bins.sum(axis=1),
    }
    out["avg_coefficient_pSv_cm2"] = out["dose_pSv"] / total_phi
    for name, (lo, hi) in BANDS.items():
        mask = (mid >= lo) & (mid < hi)
        if mask.any():
            out[f"dose_{name}_pSv"] = dose_bins[:, mask].sum(axis=1)
    return out


def summarise(results):
    """Table of mean, standard deviation, relative std and 95% interval."""
    rows = {}
    for name, x in results.items():
        mean, std = float(x.mean()), float(x.std(ddof=1))
        rows[name] = {
            "mean": mean,
            "std": std,
            "rel_std_%": 100.0 * std / mean if mean != 0 else np.nan,
            "p2.5": float(np.percentile(x, 2.5)),
            "p97.5": float(np.percentile(x, 97.5)),
        }
    return pd.DataFrame(rows).T


def rebin_matrix(old_edges, new_edges):
    """Matrix W so that new_phi = old_phi @ W.

    Assumes the fluence inside each old bin is spread evenly in ln(E), i.e.
    evenly in lethargy. Fluence is conserved wherever the new bins overlap the
    old range.
    """
    old_edges, new_edges = np.asarray(old_edges), np.asarray(new_edges)
    lo_old = np.log(old_edges[:-1])[:, None]
    hi_old = np.log(old_edges[1:])[:, None]
    lo_new = np.log(new_edges[:-1])[None, :]
    hi_new = np.log(new_edges[1:])[None, :]
    overlap = np.clip(np.minimum(hi_old, hi_new) - np.maximum(lo_old, lo_new), 0.0, None)
    return overlap / (hi_old - lo_old)


def rebin(phi, old_edges, new_edges):
    """Rebin fluence per bin onto new edges. phi can be one spectrum (1-D) or
    many (2-D, one spectrum per row)."""
    return np.asarray(phi) @ rebin_matrix(old_edges, new_edges)
