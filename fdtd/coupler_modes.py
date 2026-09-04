"""
Directional-coupler characterization by finite-element eigenmode (supermode)
analysis, using femwell (scikit-fem FEM Maxwell solver).

Method
------
Two parallel Si strip waveguides (SOI, 500 x 220 nm, SiO2 clad) support a
symmetric (even) and antisymmetric (odd) supermode with effective indices
n_even > n_odd. Power launched into one waveguide beats fully to the other
over the coupling (pi) length

        L_pi = lambda / (2 * (n_even - n_odd)),

and the cross-port power after a coupler of physical length L is

        P_cross(L) = sin^2( pi * L / (2 * L_pi) )
                   = sin^2( pi * (n_even - n_odd) * L / lambda ).

This is the rigorous eigenmode-expansion picture of a directional coupler and
is the standard method for coupler design (FDTD is the time-domain
cross-check, provided separately as a Meep script).

Outputs (written to results/):
    coupler_modes.png     : the two supermode |E| profiles at nominal gap
    coupler_sweep.png     : L_pi and cross-power vs gap and vs wavelength
    coupler_fdtd_backed.json : extracted n_eff, L_pi, and a fitted CMT model
                               ready to drop into photonic_mesh.components
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import box
from shapely.ops import unary_union
from collections import OrderedDict
from femwell.maxwell.waveguide import compute_modes
from femwell.mesh import mesh_from_OrderedDict

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Material / geometry constants (SOI, telecom C-band)
# ----------------------------------------------------------------------
N_SI = 3.4757      # crystalline Si @ 1550 nm
N_OX = 1.4440      # SiO2 @ 1550 nm
W_CORE = 0.50      # waveguide width  [um]
H_CORE = 0.22      # waveguide height [um]
LAM0 = 1.55        # design wavelength [um]

# Simulation window
XSPAN = 4.0        # total width  [um]
YSPAN = 3.0        # total height [um]


def build_coupler_mesh(gap_um, resolution=0.020, single=False):
    """
    Build a meshed cross-section: SiO2 background + one or two Si cores.
    Returns (mesh, subdomain->n_index dict).
    """
    clad = box(-XSPAN / 2, -YSPAN / 2, XSPAN / 2, YSPAN / 2)
    if single:
        cores = [box(-W_CORE / 2, -H_CORE / 2, W_CORE / 2, H_CORE / 2)]
    else:
        # two cores centered symmetrically about x=0 with edge-to-edge `gap`
        x0 = gap_um / 2.0
        cores = [
            box(x0, -H_CORE / 2, x0 + W_CORE, H_CORE / 2),
            box(-(x0 + W_CORE), -H_CORE / 2, -x0, H_CORE / 2),
        ]
    core_union = unary_union(cores)

    polys = OrderedDict(
        core=core_union,
        clad=clad,
    )
    resolutions = {
        "core": {"resolution": resolution, "distance": 0.5},
    }
    mesh = mesh_from_OrderedDict(
        polys, resolutions, default_resolution_max=0.15, filename="/tmp/mesh.msh"
    )
    return mesh


def solve_modes(gap_um, wavelength_um, num_modes=2, single=False,
                resolution=0.020):
    mesh = build_coupler_mesh(gap_um, resolution=resolution, single=single)
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio

    m = from_meshio(mesh)
    basis0 = Basis(m, ElementTriP0())
    eps = basis0.zeros() + N_OX ** 2
    eps[basis0.get_dofs(elements="core")] = N_SI ** 2

    modes = compute_modes(
        basis0, eps, wavelength=wavelength_um,
        num_modes=num_modes, order=2,
    )
    return modes


def coupling_length(gap_um, wavelength_um, resolution=0.020):
    """L_pi = lambda / (2 (n_even - n_odd)) from the two supermodes [um]."""
    modes = solve_modes(gap_um, wavelength_um, num_modes=2,
                        resolution=resolution)
    n = sorted((np.real(m.n_eff) for m in modes), reverse=True)
    n_even, n_odd = n[0], n[1]
    dn = n_even - n_odd
    Lpi = wavelength_um / (2.0 * dn)
    return Lpi, n_even, n_odd


def cross_power(L_um, gap_um, wavelength_um, Lpi=None):
    if Lpi is None:
        Lpi, _, _ = coupling_length(gap_um, wavelength_um)
    return np.sin(np.pi * L_um / (2.0 * Lpi)) ** 2


if __name__ == "__main__":
    print("=== Sanity check: single 500x220 nm Si strip waveguide ===")
    single = solve_modes(gap_um=0.0, wavelength_um=LAM0, num_modes=2,
                         single=True)
    neff_single = np.real(single[0].n_eff)
    print(f"  fundamental n_eff = {neff_single:.4f}  (expect ~2.3-2.5 TE0)")
    assert 2.0 < neff_single < 2.7, "single-wg n_eff out of expected range"
    print("  PASS\n")

    # ---- supermodes at nominal gap, plot profiles ----
    GAP0 = 0.20
    print(f"=== Supermodes of the coupler at gap = {GAP0*1000:.0f} nm ===")
    modes = solve_modes(GAP0, LAM0, num_modes=2)
    ns = sorted(((np.real(m.n_eff), m) for m in modes),
                key=lambda t: t[0], reverse=True)
    n_even, n_odd = ns[0][0], ns[1][0]
    Lpi0 = LAM0 / (2 * (n_even - n_odd))
    print(f"  n_even = {n_even:.5f}")
    print(f"  n_odd  = {n_odd:.5f}")
    print(f"  Delta n = {n_even - n_odd:.5f}")
    print(f"  L_pi (full crossover) = {Lpi0:.2f} um")
    print(f"  L_3dB (50:50)         = {Lpi0/2:.2f} um\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (neff, mode), name in zip(axes, ns, ["even (symmetric)",
                                                  "odd (antisymmetric)"]):
        # signed dominant (Ex, the TE component) reveals the even/odd sign
        # structure between the two cores -- the physics intensity plots hide.
        mode.plot_component("E", "x", part="real", ax=ax, colorbar=False)
        ax.set_title(f"{name}\n$n_{{eff}}$ = {neff:.4f}")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.7, 0.7)
        ax.set_aspect("equal")
    fig.suptitle("Directional-coupler supermodes "
                 f"(500×220 nm Si, {GAP0*1000:.0f} nm gap, 1550 nm)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/coupler_modes.png", dpi=150)
    print(f"  wrote {OUT}/coupler_modes.png\n")

    # ---- gap sweep at 1550 nm ----
    print("=== Gap sweep @ 1550 nm ===")
    gaps = np.array([0.15, 0.18, 0.20, 0.22, 0.25, 0.30])
    Lpi_gap, dn_gap = [], []
    for g in gaps:
        Lpi, ne, no = coupling_length(g, LAM0)
        Lpi_gap.append(Lpi); dn_gap.append(ne - no)
        print(f"  gap {g*1000:5.0f} nm : dn = {ne-no:.5f}, L_pi = {Lpi:6.2f} um")
    Lpi_gap = np.array(Lpi_gap)

    # ---- wavelength sweep at nominal gap ----
    print("\n=== Wavelength sweep @ gap = 200 nm ===")
    lams = np.array([1.50, 1.52, 1.55, 1.58, 1.60])
    Lpi_lam = []
    for l in lams:
        Lpi, ne, no = coupling_length(GAP0, l)
        Lpi_lam.append(Lpi)
        print(f"  lam {l*1000:.0f} nm : dn = {ne-no:.5f}, L_pi = {Lpi:6.2f} um")
    Lpi_lam = np.array(Lpi_lam)

    # design a 50:50 coupler: choose L = L_3dB at nominal gap/wavelength
    L_3dB = Lpi0 / 2.0

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
    ax2[0].plot(gaps * 1000, Lpi_gap, "o-", color="tab:blue")
    ax2[0].set_xlabel("gap [nm]"); ax2[0].set_ylabel("$L_\\pi$ [µm]")
    ax2[0].set_title("Coupling length vs gap (1550 nm)")
    ax2[0].grid(alpha=0.3)
    # cross power of the FIXED 50:50-designed coupler vs wavelength
    Pcross_lam = np.sin(np.pi * L_3dB / (2 * Lpi_lam)) ** 2
    ax2[1].plot(lams * 1000, 100 * Pcross_lam, "s-", color="tab:purple")
    ax2[1].axhline(50, ls="--", c="gray")
    ax2[1].set_xlabel("wavelength [nm]")
    ax2[1].set_ylabel("cross-port power [%]")
    ax2[1].set_title(f"50:50 coupler (L={L_3dB:.2f} µm) vs wavelength")
    ax2[1].grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/coupler_sweep.png", dpi=150)
    print(f"\n  wrote {OUT}/coupler_sweep.png")

    # ---- export FEM-backed model for photonic_mesh.components ----
    # Fit dn(gap) exponential and dn(lambda) linear for a compact drop-in.
    A = np.polyfit(gaps, np.log(np.array(dn_gap)), 1)  # ln dn = A0*gap + A1
    dn0 = float(np.exp(A[1] + A[0] * GAP0))
    gap_decay_per_um = float(A[0])
    B = np.polyfit(lams, np.array(Lpi_lam), 1)
    export = {
        "method": "femwell FEM supermode (EME) analysis",
        "geometry": {"width_um": W_CORE, "height_um": H_CORE,
                     "n_si": N_SI, "n_ox": N_OX},
        "nominal": {"gap_um": GAP0, "wavelength_um": LAM0,
                    "n_even": n_even, "n_odd": n_odd,
                    "delta_n": n_even - n_odd,
                    "L_pi_um": Lpi0, "L_3dB_um": L_3dB},
        "gap_sweep": {"gap_um": gaps.tolist(),
                      "delta_n": [float(x) for x in dn_gap],
                      "L_pi_um": Lpi_gap.tolist()},
        "wavelength_sweep": {"wavelength_um": lams.tolist(),
                             "L_pi_um": Lpi_lam.tolist()},
        "fit": {"delta_n_at_nominal": dn0,
                "gap_decay_per_um": gap_decay_per_um,
                "Lpi_vs_lambda_slope_um_per_um": float(B[0]),
                "Lpi_vs_lambda_intercept_um": float(B[1])},
    }
    with open(f"{OUT}/coupler_fem_backed.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"  wrote {OUT}/coupler_fem_backed.json")
    print("\nDONE.")
