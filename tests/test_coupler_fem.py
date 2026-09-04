"""
Guards for the FEM-backed coupler model. Runs only if the femwell-generated
data file exists (i.e. after `python fdtd/coupler_modes.py`); otherwise skips,
so the core test suite stays runnable without the FEM dependency chain.
"""
import os
import numpy as np

try:
    import pytest
except ModuleNotFoundError:
    class _P:
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                def deco(fn):
                    fn._skip = cond
                    return fn
                return deco
    pytest = _P()

from photonic_mesh.coupler_fem import get_model

_HAVE = get_model() is not None


@pytest.mark.skipif(not _HAVE, reason="FEM data not generated")
def test_fifty_fifty_at_design_wavelength():
    m = get_model()
    p = m.cross_power(1.55)
    assert abs(p - 0.5) < 1e-3, f"designed 50:50 point is off: {p}"


@pytest.mark.skipif(not _HAVE, reason="FEM data not generated")
def test_neff_ordering_and_range():
    m = get_model()
    nom = m.d["nominal"]
    assert nom["n_even"] > nom["n_odd"], "even mode must have higher n_eff"
    assert 2.0 < nom["n_odd"] < 2.7 and 2.0 < nom["n_even"] < 2.7


@pytest.mark.skipif(not _HAVE, reason="FEM data not generated")
def test_dispersion_monotonic():
    # cross power increases with wavelength for this coupler (Δn grows)
    m = get_model()
    ps = [m.cross_power(l) for l in (1.50, 1.53, 1.55, 1.58, 1.60)]
    assert all(b > a for a, b in zip(ps, ps[1:])), "expected monotone rise"


@pytest.mark.skipif(not _HAVE, reason="FEM data not generated")
def test_coupling_falls_with_gap():
    # wider gap -> longer L_pi -> less cross coupling at fixed length
    m = get_model()
    assert m.L_pi(1.55, 0.30) > m.L_pi(1.55, 0.15)


if __name__ == "__main__":
    if not _HAVE:
        print("SKIP: FEM data not generated (run fdtd/coupler_modes.py first)")
    else:
        fns = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v)]
        for f in fns:
            f(); print(f"  PASS {f.__name__}")
        print(f"\n{len(fns)} FEM checks passed.")
