"""Fold binned neutron spectra with fluence-to-H*(10) conversion coefficients.

Units: energy in MeV, fluence in cm^-2, coefficients in pSv cm^2, dose in pSv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.spectrum import bin_midpoints

# Energy bands in MeV (thermal < 0.5 eV, epithermal 0.5 eV - 100 keV,
# fast 100 keV - 20 MeV, high > 20 MeV)
BANDS = {
    "thermal": (0.0, 5e-7),
    "epithermal": (5e-7, 0.1),
    "fast": (0.1, 20.0),
    "high": (20.0, np.inf),
}

DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "data" / "h10_neutron_coefficients.csv"


def load_coefficients(path=DEFAULT_TABLE):
    """Read the coefficient table. Returns (energies_MeV, h10_pSv_cm2)."""
    df = pd.read_csv(path).sort_values("energy_MeV")
    return df["energy_MeV"].to_numpy(), df["h10_pSv_cm2"].to_numpy()


def interpolate_coefficients(E, energies, coeffs):
    """Log-log interpolation of the table. Raises if E is outside the table."""
    E = np.asarray(E, dtype=float)
    if E.min() < energies.min() or E.max() > energies.max():
        raise ValueError(
            f"Energy range {E.min():.3g}-{E.max():.3g} MeV is outside the "
            f"table range {energies.min():.3g}-{energies.max():.3g} MeV."
        )
    return np.exp(np.interp(np.log(E), np.log(energies), np.log(coeffs)))


def dose_per_bin(edges, phi_bins, energies, coeffs):
    """Dose (pSv) in each bin: bin fluence times the coefficient at bin midpoint."""
    h = interpolate_coefficients(bin_midpoints(edges), energies, coeffs)
    return phi_bins * h


def total_dose(edges, phi_bins, energies, coeffs):
    return float(np.sum(dose_per_bin(edges, phi_bins, energies, coeffs)))


def dose_by_band(edges, phi_bins, energies, coeffs):
    """Dose split into thermal / epithermal / fast / high bands, plus total."""
    d = dose_per_bin(edges, phi_bins, energies, coeffs)
    mid = bin_midpoints(edges)
    out = {name: float(d[(mid >= lo) & (mid < hi)].sum())
           for name, (lo, hi) in BANDS.items()}
    out["total"] = float(d.sum())
    return out


def spectrum_averaged_coefficient(edges, phi_bins, energies, coeffs):
    """Total dose divided by total fluence (pSv cm^2)."""
    return total_dose(edges, phi_bins, energies, coeffs) / float(np.sum(phi_bins))