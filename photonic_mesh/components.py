"""
Physics-based component models for a programmable photonic mesh.

Conventions
-----------
- All 2x2 blocks act on the mode vector (a_top, a_bottom)^T, where each entry
  is a complex field amplitude (|a|^2 is power, arg(a) is optical phase).
- A directional coupler with coupling angle ``theta_c`` has transfer matrix
      C(theta_c) = [[ cos(theta_c),  i sin(theta_c)],
                    [ i sin(theta_c), cos(theta_c) ]]
  so theta_c = pi/4 gives an ideal 50:50 splitter (cross power = sin^2 = 0.5).
  The factor of ``i`` on the cross terms is the pi/2 through-vs-coupled phase
  required by unitarity (energy conservation) -- not cosmetic.
- Loss is applied as a scalar *amplitude* factor a = 10^(-loss_db/20) per
  element. Uniform loss commutes with the unitary, so it reduces transmission
  without distorting the implemented matrix; path-dependent (differential)
  loss across a mesh does distort it, which the fidelity metric captures.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Directional coupler
# ----------------------------------------------------------------------

def coupler_angle_from_splitting(power_cross: float) -> float:
    """Coupler angle theta_c such that cross-port power = ``power_cross``."""
    return float(np.arcsin(np.sqrt(np.clip(power_cross, 0.0, 1.0))))


def dc_cross_coupling(wavelength_um: float,
                      gap_um: float = 0.20,
                      length_um: float = 14.0,
                      kappa0_per_um: float = 0.0561,
                      dkappa_dlam_per_nm: float = 0.0028,
                      dkappa_dgap_per_nm: float = -0.045,
                      lam0_um: float = 1.55) -> float:
    """
    Coupled-mode-theory model of a Si directional coupler's cross-port power.

    Two waveguides brought close enough for their evanescent fields to overlap
    exchange power along their length as ``sin^2(kappa * L)``. The coupling
    strength ``kappa`` is linearized about a nominal 50:50 operating point:

        kappa = kappa0 * (1 + dkappa_dlam_per_nm * (lam - lam0)[nm])
                       * (1 + dkappa_dgap_per_nm * (gap - 0.20)[nm])

    Defaults give a device that is 50:50 at 1550 nm (kappa0 * L = pi/4) with
    representative sensitivities for a ~500x220 nm strip-waveguide Si coupler
    (~+0.28 %/nm in wavelength, ~-4.5 %/nm in gap). These coefficients are
    illustrative, NOT silicon-calibrated -- replace this whole function with
    FDTD-extracted S-parameters for a specific process (see README).

    Returns cross-port power in [0, 1].
    """
    dlam_nm = (wavelength_um - lam0_um) * 1000.0
    dgap_nm = (gap_um - 0.20) * 1000.0
    kappa = kappa0_per_um * (1.0 + dkappa_dlam_per_nm * dlam_nm) \
                          * (1.0 + dkappa_dgap_per_nm * dgap_nm)
    kappa = max(kappa, 0.0)
    return float(np.sin(kappa * length_um) ** 2)


def coupler_matrix(theta_c: float, loss_db: float = 0.0) -> np.ndarray:
    """2x2 directional-coupler transfer matrix with insertion loss."""
    a = 10 ** (-loss_db / 20.0)          # amplitude factor (power ~ a^2)
    c, s = np.cos(theta_c), np.sin(theta_c)
    return a * np.array([[c, 1j * s],
                         [1j * s, c]], dtype=complex)


# ----------------------------------------------------------------------
# Phase shifter
# ----------------------------------------------------------------------

def phase_matrix(phi_top: float, phi_bot: float = 0.0,
                 loss_db: float = 0.0) -> np.ndarray:
    """
    2x2 phase-shifter block (e.g. a thermo-optic heater on one arm).

    Diagonal because a phase shifter does not mix the two waveguides -- it
    only retards them independently.
    """
    a = 10 ** (-loss_db / 20.0)
    return a * np.diag([np.exp(1j * phi_top), np.exp(1j * phi_bot)])


# ----------------------------------------------------------------------
# Mach-Zehnder interferometer
# ----------------------------------------------------------------------

def mzi_matrix(theta: float, phi: float,
               theta_c1: float = np.pi / 4,
               theta_c2: float = np.pi / 4,
               coupler_loss_db: float = 0.0,
               ps_loss_db: float = 0.0) -> np.ndarray:
    """
    2x2 MZI transfer matrix in the Clements convention:

        U = C2 . P(theta) . C1 . P(phi)

    Read right-to-left in the order light experiences the elements.

    theta    : internal differential phase (sets the MZI splitting ratio)
    phi      : external phase on the top input arm (sets output phase)
    theta_c1 : first coupler angle  (pi/4 = ideal 50:50)
    theta_c2 : second coupler angle (pi/4 = ideal 50:50)

    Passing theta_c1 != theta_c2 != pi/4 is how fabrication error enters the
    model: a real MZI has two independently imperfect couplers.
    """
    C1 = coupler_matrix(theta_c1, coupler_loss_db)
    C2 = coupler_matrix(theta_c2, coupler_loss_db)
    Pth = phase_matrix(theta, 0.0, ps_loss_db)
    Pph = phase_matrix(phi, 0.0, ps_loss_db)
    return C2 @ Pth @ C1 @ Pph


def embed_2x2(U2: np.ndarray, m: int, N: int) -> np.ndarray:
    """
    Embed a 2x2 block acting on modes (m, m+1) into an NxN identity.

    A mesh MZI touches only two adjacent waveguides; the other N-2 modes pass
    through untouched. This lifts a local 2x2 device to a global NxN operator.
    """
    T = np.eye(N, dtype=complex)
    T[m:m + 2, m:m + 2] = U2
    return T
