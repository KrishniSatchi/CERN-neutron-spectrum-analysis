"""Dose by energy band for the synthetic moderated example field."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.spectrum import log_bins, spectrum_from_function, normalise
from src.example_fields import moderated_field_pdf
from src.dose import load_coefficients, dose_by_band, spectrum_averaged_coefficient

ROOT = Path(__file__).resolve().parent.parent

energies, coeffs = load_coefficients()
edges = log_bins(energies.min(), min(energies.max(), 20.0), 600)   # stay inside the table
phi = normalise(spectrum_from_function(edges, moderated_field_pdf))

bands = dose_by_band(edges, phi, energies, coeffs)
avg = spectrum_averaged_coefficient(edges, phi, energies, coeffs)

print("Dose per unit fluence (pSv per neutron/cm^2), synthetic example field")
for name, value in bands.items():
    print(f"  {name:11s} {value:10.3f}  ({value / bands['total']:.1%})")
print(f"Spectrum-averaged coefficient: {avg:.1f} pSv cm^2")

names = [n for n in bands if n != "total"]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.bar(names, [bands[n] for n in names])
ax.set_ylabel("Dose per unit fluence (pSv per cm$^{-2}$)")
ax.set_title("Ambient dose equivalent by energy band (synthetic field)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "dose_by_band.png", dpi=300)
