"""
Project study: 4x4 Clements-mesh programmable photonic processor.

Generates:
  results/fig1_fidelity_vs_errors.png : fidelity vs coupler-splitting error
                                        and vs phase-setting error
  results/fig2_calibration.png        : fidelity before/after on-chip phase
                                        self-calibration (fixed coupler error)
  results/fig3_wavelength.png         : coupler dispersion -> mesh fidelity(lambda)
  results/summary.txt                 : headline numbers

Error model
-----------
* Coupler splitting error: cross power P = sin^2(theta_c); near 50:50,
  d(sin^2 theta_c)/dtheta_c = sin(2 theta_c) = 1, so a +/-x% power imbalance
  maps to ~x/100 rad of coupler-angle error. theta_c ~ N(pi/4, sigma_c),
  drawn independently for both couplers of every MZI (fab error is local).
* Phase error: additive N(0, sigma_p) on every programmed theta and phi
  (DAC quantization + thermal crosstalk + calibration residue).
* Loss: 0.10 dB / coupler, 0.05 dB / phase-shifter section (representative
  good Si numbers), reported as mean mesh transmission.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from scipy.optimize import minimize

from photonic_mesh import (
    decompose, build_mesh, fidelity, transmission_db,
    dc_cross_coupling, coupler_angle_from_splitting,
)
from photonic_mesh.coupler_fem import get_model

_FEM = get_model()
if _FEM is not None:
    coupler_cross = lambda l: _FEM.cross_power(l)
    COUPLER_SOURCE = "femwell FEM supermode analysis"
else:
    coupler_cross = lambda l: dc_cross_coupling(l)
    COUPLER_SOURCE = "analytic CMT placeholder (FEM data not found)"

OUT = "results"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)
N = 4
N_TRIALS = 300
N_UNITARIES = 20
COUPLER_LOSS_DB = 0.10
PS_LOSS_DB = 0.05

targets = [unitary_group.rvs(N, random_state=rng) for _ in range(N_UNITARIES)]
programs = [decompose(U) for U in targets]
n_mzi = len(programs[0].mzis)
print(f"{n_mzi} MZIs in the 4x4 Clements mesh (expected N(N-1)/2 = 6)")
print(f"coupler model: {COUPLER_SOURCE}")


# ----------------------------------------------------------------------
# Study 1: fidelity vs coupler error and vs phase error
# ----------------------------------------------------------------------
sigma_c_list = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
sigma_p_list = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.20])


def mc_fidelity(sigma_c, sigma_p, trials=N_TRIALS):
    Fs = np.empty(trials)
    for t in range(trials):
        i = t % N_UNITARIES
        ca = (np.pi / 4 + rng.normal(0, sigma_c, size=(n_mzi, 2))
              if sigma_c else None)
        pe = rng.normal(0, sigma_p, size=(n_mzi, 2)) if sigma_p else None
        Ua = build_mesh(programs[i], coupler_angles=ca, phase_errors=pe,
                        coupler_loss_db=COUPLER_LOSS_DB, ps_loss_db=PS_LOSS_DB)
        Fs[t] = fidelity(targets[i], Ua)
    return Fs


F_vs_c = np.array([mc_fidelity(sc, 0.0) for sc in sigma_c_list])
F_vs_p = np.array([mc_fidelity(0.0, sp) for sp in sigma_p_list])
F_headline = mc_fidelity(0.02, 0.05)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].errorbar(100 * sigma_c_list, F_vs_c.mean(1), yerr=F_vs_c.std(1),
               fmt="o-", capsize=3, color="tab:blue")
ax[0].set_xlabel("coupler splitting error σ  [% power imbalance]")
ax[0].set_ylabel("unitary fidelity")
ax[0].set_title("Fidelity vs coupler fab error (σ_phase = 0)")
ax[0].grid(alpha=0.3)
ax[1].errorbar(sigma_p_list, F_vs_p.mean(1), yerr=F_vs_p.std(1),
               fmt="s-", capsize=3, color="tab:red")
ax[1].set_xlabel("phase-setting error σ  [rad]")
ax[1].set_ylabel("unitary fidelity")
ax[1].set_title("Fidelity vs phase error (ideal couplers)")
ax[1].grid(alpha=0.3)
fig.suptitle("4×4 Clements mesh — Monte Carlo fab tolerance "
             f"({N_TRIALS} meshes/point, {N_UNITARIES} Haar-random targets)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_fidelity_vs_errors.png", dpi=160)


# ----------------------------------------------------------------------
# Study 2: self-calibration against fixed (measured) coupler errors
# ----------------------------------------------------------------------
def calibrate(prog, U_t, coupler_angles):
    n = len(prog.mzis)

    def cost(x):
        pe = x[:2 * n].reshape(n, 2)
        oc = x[2 * n:]
        Ua = build_mesh(prog, coupler_angles=coupler_angles, phase_errors=pe,
                        coupler_loss_db=COUPLER_LOSS_DB, ps_loss_db=PS_LOSS_DB)
        Ua = np.diag(np.exp(1j * oc)) @ Ua
        return -fidelity(U_t, Ua)

    res = minimize(cost, np.zeros(2 * n + N), method="L-BFGS-B")
    return -res.fun


SIGMA_C_CAL = 0.02
F_before, F_after = [], []
for t in range(80):
    i = t % N_UNITARIES
    ca = np.pi / 4 + rng.normal(0, SIGMA_C_CAL, size=(n_mzi, 2))
    Ua = build_mesh(programs[i], coupler_angles=ca,
                    coupler_loss_db=COUPLER_LOSS_DB, ps_loss_db=PS_LOSS_DB)
    F_before.append(fidelity(targets[i], Ua))
    F_after.append(calibrate(programs[i], targets[i], ca))
F_before, F_after = np.array(F_before), np.array(F_after)

fig2, ax2 = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(min(F_before.min(), 0.985), 1.0001, 40)
ax2.hist(F_before, bins=bins, alpha=0.6, color="tab:red",
         label=f"uncalibrated (mean {F_before.mean():.4f})")
ax2.hist(F_after, bins=bins, alpha=0.6, color="tab:green",
         label=f"phase-calibrated (mean {F_after.mean():.4f})")
ax2.set_xlabel("unitary fidelity")
ax2.set_ylabel("count")
ax2.set_title(f"Self-calibration vs ±{100*SIGMA_C_CAL:.0f}% coupler errors "
              "(4×4 mesh, 80 chips)")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig2_calibration.png", dpi=160)


# ----------------------------------------------------------------------
# Study 3: wavelength dependence from coupler dispersion
# ----------------------------------------------------------------------
lams = np.linspace(1.50, 1.60, 41)
split = np.array([coupler_cross(l) for l in lams])
F_lam = []
for l in lams:
    th_c = coupler_angle_from_splitting(coupler_cross(l))
    ca = np.full((n_mzi, 2), th_c)
    F_lam.append(np.mean([
        fidelity(targets[i],
                 build_mesh(programs[i], coupler_angles=ca,
                            coupler_loss_db=COUPLER_LOSS_DB,
                            ps_loss_db=PS_LOSS_DB))
        for i in range(N_UNITARIES)]))

fig3, ax3 = plt.subplots(1, 2, figsize=(11, 4.2))
ax3[0].plot(1000 * lams, 100 * split, color="tab:blue")
ax3[0].axhline(50, ls="--", c="gray")
ax3[0].set_xlabel("wavelength [nm]")
ax3[0].set_ylabel("cross-port power [%]")
ax3[0].set_title("Directional-coupler dispersion (FEM supermode)")
ax3[0].grid(alpha=0.3)
ax3[1].plot(1000 * lams, F_lam, color="tab:purple")
ax3[1].set_xlabel("wavelength [nm]")
ax3[1].set_ylabel("mean unitary fidelity")
ax3[1].set_title("Mesh fidelity vs wavelength (programmed at 1550 nm)")
ax3[1].grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig(f"{OUT}/fig3_wavelength.png", dpi=160)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
U0 = build_mesh(programs[0], coupler_loss_db=COUPLER_LOSS_DB,
                ps_loss_db=PS_LOSS_DB)
lines = [
    "4x4 Clements-mesh programmable processor - study summary",
    "=" * 60,
    f"Coupler model                             : {COUPLER_SOURCE}",
    f"MZIs in mesh                              : {n_mzi}",
    f"Mesh transmission (0.10 dB/DC, 0.05 dB/PS): {transmission_db(U0):.2f} dB",
    "",
    "Fidelity vs coupler splitting error (sigma_phase = 0):",
    *[f"  sigma_c = {100*sc:4.1f}%  ->  F = {F.mean():.5f} +/- {F.std():.5f}"
      for sc, F in zip(sigma_c_list, F_vs_c)],
    "",
    "Fidelity vs phase error (ideal couplers):",
    *[f"  sigma_p = {sp:4.2f} rad ->  F = {F.mean():.5f} +/- {F.std():.5f}"
      for sp, F in zip(sigma_p_list, F_vs_p)],
    "",
    "Headline combined case (sigma_c = 2%, sigma_p = 0.05 rad):",
    f"  F = {F_headline.mean():.5f} +/- {F_headline.std():.5f}",
    "",
    "Self-calibration vs fixed +/-2% coupler errors (80 chips):",
    f"  uncalibrated  F = {F_before.mean():.5f} (worst {F_before.min():.5f})",
    f"  calibrated    F = {F_after.mean():.5f} (worst {F_after.min():.5f})",
    f"  mean infidelity reduced by "
    f"{(1-F_before.mean())/(1-F_after.mean()+1e-15):.0f}x",
]
summary = "\n".join(lines)
print("\n" + summary)
with open(f"{OUT}/summary.txt", "w") as f:
    f.write(summary + "\n")
print(f"\nWrote figures and summary to ./{OUT}/")
