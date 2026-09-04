"""
Clements decomposition (Clements, Humphreys, Metcalf, Kolthammer & Walmsley,
*Optica* 3, 1460 (2016)) using the *physical* MZI matrix throughout, plus a
forward mesh model with imperfections for fabrication-tolerance studies.

Physical MZI (components.mzi_matrix, ideal 50:50 couplers):

    M(theta, phi) = i e^{i theta/2} [[e^{i phi} sin(t/2),  cos(t/2)],
                                     [e^{i phi} cos(t/2), -sin(t/2)]]

This is universal on U(2) up to output phases, which the algorithm collects
into a final diagonal layer D:   U = D . prod_k T_k     (T_k = embedded MZI).

Identities used (each verified numerically to ~1e-16 in the test suite):
  Right-null:  (V M^dag)[i, m]   = 0  <=>  tan(t/2) e^{-i phi} = -V[i,m+1]/V[i,m]
  Left-null :  (M V)[m+1, c]     = 0  <=>  tan(t/2) e^{ i phi} =  V[m,c]/V[m+1,c]
  Commutation: M(t,p)^dag @ diag(d1, d2)
                 = -e^{-i t} diag(d2 e^{-i p}, d2) @ M(t, angle(d1/d2))
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .components import mzi_matrix


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class MZISetting:
    """One MZI's programmed settings; acts on adjacent modes (m, m+1)."""
    m: int
    theta: float     # internal phase (splitting ratio)
    phi: float       # external phase (output phase)


@dataclass
class MeshProgram:
    """A full mesh: ordered MZI settings + an output diagonal phase layer."""
    N: int
    mzis: List[MZISetting] = field(default_factory=list)
    output_phases: np.ndarray | None = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _embed(U2: np.ndarray, m: int, N: int) -> np.ndarray:
    T = np.eye(N, dtype=complex)
    T[m:m + 2, m:m + 2] = U2
    return T


def _solve_right_null(a: complex, b: complex) -> tuple[float, float]:
    """(theta, phi) so that right-multiplying by M(theta,phi)^dag nulls the
    element built from a = V[i, m], b = V[i, m+1]:
        tan(theta/2) e^{-i phi} = -b / a.
    """
    theta = 2.0 * np.arctan2(np.abs(b), np.abs(a))
    phi = -np.angle(-b) + np.angle(a) if abs(a) > 0 else 0.0
    return theta, phi


def _solve_left_null(a: complex, b: complex) -> tuple[float, float]:
    """(theta, phi) so that left-multiplying by M(theta,phi) nulls the
    element built from a = V[m, c], b = V[m+1, c]:
        tan(theta/2) e^{i phi} = a / b.
    """
    theta = 2.0 * np.arctan2(np.abs(a), np.abs(b))
    phi = -(np.angle(a) - np.angle(b)) if abs(b) > 0 else 0.0
    return theta, phi


# ----------------------------------------------------------------------
# Decomposition
# ----------------------------------------------------------------------

def decompose(U: np.ndarray) -> MeshProgram:
    """
    Clements decomposition of an NxN unitary into physical-MZI settings plus
    an output phase layer, such that ``build_ideal(decompose(U))`` == U.

    Strategy: eliminate off-diagonal elements one at a time by multiplying U
    on the left or right by MZI matrices (alternating, which yields the
    rectangular / balanced Clements mesh rather than the triangular Reck one).
    What remains is a diagonal D; left-side MZIs are then commuted through D so
    that all MZIs sit on one side and D becomes a single output phase layer.
    """
    U = np.array(U, dtype=complex)
    N = U.shape[0]
    assert np.allclose(U.conj().T @ U, np.eye(N), atol=1e-8), "U not unitary"

    V = U.copy()
    left_ops: list[MZISetting] = []    # applied as M V
    right_ops: list[MZISetting] = []   # applied as V M^dag

    for k in range(N - 1):
        if k % 2 == 0:
            for j in range(k + 1):
                row, col = N - 1 - j, k - j
                m = col
                th, ph = _solve_right_null(V[row, m], V[row, m + 1])
                V = V @ _embed(mzi_matrix(th, ph), m, N).conj().T
                right_ops.append(MZISetting(m, th, ph))
        else:
            for j in range(1, k + 2):
                row, col = N + j - k - 2, j - 1
                m = row - 1
                th, ph = _solve_left_null(V[m, col], V[m + 1, col])
                V = _embed(mzi_matrix(th, ph), m, N) @ V
                left_ops.append(MZISetting(m, th, ph))

    # V is now diagonal. Commute every left op through it (right-to-left).
    D = np.diag(V).astype(complex).copy()
    assert np.allclose(np.abs(D), 1.0, atol=1e-7), "nulling failed"

    new_left: list[MZISetting] = []
    for s in reversed(left_ops):
        d1, d2 = D[s.m], D[s.m + 1]
        phi_new = np.angle(d1 / d2)
        pref = -np.exp(-1j * s.theta)
        D[s.m] = pref * d2 * np.exp(-1j * s.phi)
        D[s.m + 1] = pref * d2
        new_left.append(MZISetting(s.m, s.theta, phi_new))

    # Forward model applies mzis[0] first (rightmost in the product):
    #   U = D @ M_last @ ... @ M_first
    ordered = list(right_ops) + list(new_left)
    return MeshProgram(N=N, mzis=ordered, output_phases=np.angle(D))


# ----------------------------------------------------------------------
# Forward model
# ----------------------------------------------------------------------

def build_mesh(prog: MeshProgram,
               coupler_angles: np.ndarray | None = None,
               phase_errors: np.ndarray | None = None,
               coupler_loss_db: float = 0.0,
               ps_loss_db: float = 0.0,
               phase_overrides: np.ndarray | None = None) -> np.ndarray:
    """
    Rebuild the NxN transfer matrix from a MeshProgram.

    coupler_angles  : (n_mzi, 2) physical coupler angles; None -> ideal pi/4.
    phase_errors    : (n_mzi, 2) additive (theta, phi) errors; None -> 0.
    phase_overrides : (n_mzi, 2) absolute (theta, phi) to use instead of the
                      programmed values (used by calibration); errors are
                      still added on top.
    """
    N, n = prog.N, len(prog.mzis)
    if coupler_angles is None:
        coupler_angles = np.full((n, 2), np.pi / 4)
    if phase_errors is None:
        phase_errors = np.zeros((n, 2))

    U = np.eye(N, dtype=complex)
    for i, s in enumerate(prog.mzis):
        th = phase_overrides[i, 0] if phase_overrides is not None else s.theta
        ph = phase_overrides[i, 1] if phase_overrides is not None else s.phi
        M2 = mzi_matrix(th + phase_errors[i, 0],
                        ph + phase_errors[i, 1],
                        theta_c1=coupler_angles[i, 0],
                        theta_c2=coupler_angles[i, 1],
                        coupler_loss_db=coupler_loss_db,
                        ps_loss_db=ps_loss_db)
        U = _embed(M2, s.m, N) @ U
    if prog.output_phases is not None:
        U = np.diag(np.exp(1j * prog.output_phases)) @ U
    return U


def build_ideal(prog: MeshProgram) -> np.ndarray:
    """Forward model with perfect 50:50 couplers and no loss."""
    return build_mesh(prog)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def fidelity(U_target: np.ndarray, U_actual: np.ndarray) -> float:
    """
    Loss-normalized unitary fidelity:

        F = |Tr(U_t^dag U_a)|^2 / (N * Tr(U_a^dag U_a))

    F = 1 iff U_a = c * U_t for any complex scalar c, so uniform loss and
    global phase do not count against fidelity -- only distortion of the
    implemented unitary does (e.g. path-dependent loss across the mesh).
    """
    N = U_target.shape[0]
    num = np.abs(np.trace(U_target.conj().T @ U_actual)) ** 2
    den = N * np.real(np.trace(U_actual.conj().T @ U_actual))
    return float(num / den)


def transmission_db(U_actual: np.ndarray) -> float:
    """Average power transmission of the mesh in dB (0 dB = lossless)."""
    N = U_actual.shape[0]
    t = np.real(np.trace(U_actual.conj().T @ U_actual)) / N
    return float(10 * np.log10(t))
