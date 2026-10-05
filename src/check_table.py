"""Sanity-check the typed H*(10) coefficient table."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.dose import load_coefficients, interpolate_coefficients

ROOT = Path(__file__).resolve().parent.parent

# Independent check points: (energy in MeV, h*(10) in pSv cm^2) from
# E. S. da Fonseca and W. W. Pereira, BIPM CCRI(III)/01-10, Table 2.
# Verify these against the document yourself.
REFERENCE = [
    (2.53e-8, 10.6), (0.002, 7.7), (0.025, 19.3), (0.144, 127), (0.25, 203),
    (0.565, 343), (1.2, 425), (2.5, 416), (2.8, 413), (3.2, 411),
    (5.0, 405), (14.8, 536), (19.0, 584),
]
TOLERANCE = 0.15   # flag disagreements bigger than 15%

energies, coeffs = load_coefficients()
print(f"{len(energies)} rows, energy range {energies.min():.3g} to {energies.max():.3g} MeV")

problems = []
if np.isnan(coeffs).any() or np.isnan(energies).any():
    problems.append("missing values in the table")
if np.any(np.diff(energies) <= 0):
    problems.append("duplicate energies")
if np.any(coeffs <= 0):
    problems.append("zero or negative coefficients")

print(f"\n{'E (MeV)':>10} {'reference':>10} {'your table':>11} {'ratio':>7}")
for E, ref in REFERENCE:
    if E < energies.min() or E > energies.max():
        print(f"{E:10.3g} {ref:10.1f}   (outside your table's range)")
        continue
    mine = float(interpolate_coefficients(E, energies, coeffs))
    ratio = mine / ref
    flag = ""
    if abs(ratio - 1) > TOLERANCE:
        flag = "  <-- CHECK"
        problems.append(f"{E:.3g} MeV: your table gives {mine:.1f}, reference {ref}")
    print(f"{E:10.3g} {ref:10.1f} {mine:11.1f} {ratio:7.2f}{flag}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(energies, coeffs, "o-", ms=3, label="Your table")
ax.loglog(*zip(*REFERENCE), "rx", label="Second-source check points")
ax.set_xlabel("Neutron energy (MeV)")
ax.set_ylabel("h*(10) (pSv cm$^2$)")
ax.set_title("Fluence-to-ambient-dose-equivalent coefficients: table check")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "h10_table_check.png", dpi=200)

print("\n" + ("PROBLEMS FOUND:\n  " + "\n  ".join(problems) if problems else "No problems found."))