"""
Verification suite for the Clements decomposition and its underlying
identities. Run with:  pytest -q
"""
import numpy as np
from scipy.stats import unitary_group

try:
    import pytest
except ModuleNotFoundError:  # allow `python3 tests/test_clements.py` w/o pytest
    class _Pytest:
        class mark:
            @staticmethod
            def parametrize(name, values):
                def deco(fn):
                    fn._params = (name, values)
                    return fn
                return deco
    pytest = _Pytest()

from photonic_mesh.components import mzi_matrix
from photonic_mesh.clements import (
    decompose, build_ideal, fidelity,
    _solve_right_null, _solve_left_null, _embed,
)


# ----------------------------------------------------------------------
# The two nulling identities, in isolation
# ----------------------------------------------------------------------

def test_right_null_zeros_element():
    rng = np.random.default_rng(1)
    for _ in range(200):
        V = unitary_group.rvs(2, random_state=rng)
        th, ph = _solve_right_null(V[0, 0], V[0, 1])
        out = V @ mzi_matrix(th, ph).conj().T
        assert abs(out[0, 0]) < 1e-12


def test_left_null_zeros_element():
    rng = np.random.default_rng(2)
    for _ in range(200):
        V = unitary_group.rvs(2, random_state=rng)
        th, ph = _solve_left_null(V[0, 0], V[1, 0])
        out = mzi_matrix(th, ph) @ V
        assert abs(out[1, 0]) < 1e-12


# ----------------------------------------------------------------------
# The diagonal-commutation identity used to collect the output phases
# ----------------------------------------------------------------------

def test_commutation_identity():
    rng = np.random.default_rng(3)
    for _ in range(200):
        th = rng.uniform(0, np.pi)
        ph = rng.uniform(-np.pi, np.pi)
        d1 = np.exp(1j * rng.uniform(-np.pi, np.pi))
        d2 = np.exp(1j * rng.uniform(-np.pi, np.pi))
        lhs = mzi_matrix(th, ph).conj().T @ np.diag([d1, d2])
        phi_p = np.angle(d1 / d2)
        rhs = (-np.exp(-1j * th)
               * np.diag([d2 * np.exp(-1j * ph), d2])
               @ mzi_matrix(th, phi_p))
        assert np.max(np.abs(lhs - rhs)) < 1e-12


# ----------------------------------------------------------------------
# Full decomposition exactness
# ----------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3, 4, 5, 6, 8])
def test_random_unitary_reconstruction(N):
    rng = np.random.default_rng(100 + N)
    for _ in range(40):
        U = unitary_group.rvs(N, random_state=rng)
        F = fidelity(U, build_ideal(decompose(U)))
        assert F > 1 - 1e-10


def test_mzi_count_is_triangular():
    rng = np.random.default_rng(7)
    for N in (2, 3, 4, 5, 6):
        prog = decompose(unitary_group.rvs(N, random_state=rng))
        assert len(prog.mzis) == N * (N - 1) // 2


def test_dft_reconstruction():
    N = 4
    dft = np.exp(2j * np.pi * np.outer(range(N), range(N)) / N) / np.sqrt(N)
    assert fidelity(dft, build_ideal(decompose(dft))) > 1 - 1e-10


def test_identity_and_swap():
    # Identity
    assert fidelity(np.eye(4), build_ideal(decompose(np.eye(4)))) > 1 - 1e-10
    # A permutation (full 4-cycle) is unitary and a good structured check
    P = np.roll(np.eye(4), 1, axis=0).astype(complex)
    assert fidelity(P, build_ideal(decompose(P))) > 1 - 1e-10


if __name__ == "__main__":
    # Minimal stdlib runner so `python3 tests/test_clements.py` works without
    # pytest. GitHub Actions still runs the same functions via pytest.
    import inspect
    fns = [f for name, f in sorted(globals().items())
           if name.startswith("test_") and callable(f)]
    passed = 0
    for f in fns:
        params = getattr(f, "_params", None)
        if params:
            name, values = params
            for v in values:
                f(v)
                print(f"  PASS {f.__name__}({name}={v})")
                passed += 1
        else:
            f()
            print(f"  PASS {f.__name__}")
            passed += 1
    print(f"\n{passed} checks passed.")
