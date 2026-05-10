"""
save_figures.py — Re-generate all report figures as static PNG files.

Run this script AFTER executing code.ipynb in full (all variables in memory),
or execute each cell block inside the notebook.

Output directory: results/figures/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, scipy.io, scipy.linalg, scipy.signal
from scipy.signal import butter as _butter

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
FIG_DIR= os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Reload data (standalone) ──────────────────────────────────────────────────
mat  = scipy.io.loadmat(os.path.join(ROOT, "models", "AC_model.mat"))
A    = mat["A"];  B = mat["B"];  C = mat["C"];  D = mat["D"]
VTAS = mat["VTAS"].item()

from src.config  import L, USIGMA, TAU, IN_WIND, OUT_MX_ROOT, OUT_SENSOR
from src.config  import IN_INNER_AILERON, IN_OUTER_AILERON, IN_ELEVATOR
from src.config  import LAW_GAIN, LAW_FILTER_ORDER, LAW_CUTOFF_HZ
from src.modal   import freq_damp
from src.transfer_functions import compute_tf
from src.turbulence         import von_karman_psd
from src.psd_response       import output_psd, rms_from_psd
from src.control_law        import law_freq_response, build_closed_loop

freq    = np.arange(0.001, 5.0, 0.01)
omega   = 2 * np.pi * freq
freq_hz, zeta, eigenvalues = freq_damp(A)
TF      = compute_tf(A, B, C, D, omega)
phi_vk  = von_karman_psd(omega, VTAS, L, USIGMA)
S_y     = output_psd(TF, phi_vk, IN_WIND, VTAS)

STYLE = dict(dpi=150, bbox_inches="tight")

# helper: signed & normalised mode shape from complex TF column
def _signed_shape(TF_col):
    """Phase-align complex TF column to real, then normalise to +/-1."""
    idx_max = np.argmax(np.abs(TF_col))
    aligned = np.real(TF_col * np.exp(-1j * np.angle(TF_col[idx_max])))
    return aligned / np.max(np.abs(aligned))

# Flex mode 1 & 2 frequencies (common to figs 1, 2, bode)
f1_flex = freq_hz[(freq_hz > 0.5) & (np.abs(zeta) < 95)][0]
f2_flex = freq_hz[(freq_hz > 0.5) & (np.abs(zeta) < 95)][2]   # skip conjugate

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1a -- Frequency vs Damping scatter (matches notebook plot)
# ─────────────────────────────────────────────────────────────────────────────
mask_show = (freq_hz > 0.01) & (np.abs(zeta) < 50)   # exclude aero lag (zeta~100%)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(freq_hz[mask_show], zeta[mask_show],
           c="#1a1a6e", s=80, zorder=3)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping (%)")
ax.set_title("Frequencies and Damping of Normal Modes")
ax.set_xlim([0, 5]); ax.set_ylim([0, 55])
ax.grid(True, alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1a_modal_scatter.png"), **STYLE)
plt.close(fig)
print("Saved fig1a_modal_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1b -- Eigenvalue map (complex plane)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
mask_rigid = freq_hz < 0.5
mask_flex  = (freq_hz >= 0.5) & (np.abs(zeta) < 95)
mask_lag   = np.abs(zeta) >= 95
ax.scatter(np.real(eigenvalues[mask_rigid]), np.imag(eigenvalues[mask_rigid]),
           c="tab:gray",   marker="s", s=60,  label="Rigid-body / integrator")
ax.scatter(np.real(eigenvalues[mask_flex]),  np.imag(eigenvalues[mask_flex]),
           c="tab:blue",   marker="o", s=80,  label="Flexible modes")
ax.scatter(np.real(eigenvalues[mask_lag]),   np.imag(eigenvalues[mask_lag]),
           c="tab:orange", marker="^", s=60,  label="Aero lag states")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axhline(0, color="k", lw=0.8, ls="-")
ax.set_xlabel("Re(lambda)  [rad/s]")
ax.set_ylabel("Im(lambda)  [rad/s]")
ax.set_title("Eigenvalue Map -- Open-loop Plant")
ax.legend(); ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1b_eigenvalues.png"), **STYLE)
plt.close(fig)
print("Saved fig1b_eigenvalues.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 -- Signed mode shapes (Z-acc at stations A-F) for modes 1 and 2
# ─────────────────────────────────────────────────────────────────────────────
stations    = ["A", "B", "C", "D", "E", "F"]
out_indices = [0, 1, 2, 3, 7, 8]          # outputs 1-4, 8-9 (0-based)

k1 = np.argmin(np.abs(freq - f1_flex))
k2 = np.argmin(np.abs(freq - f2_flex))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, k, f_mode in zip(axes, [k1, k2], [f1_flex, f2_flex]):
    z_mode = zeta[np.argmin(np.abs(freq_hz - f_mode))]
    shape  = _signed_shape(TF[out_indices, IN_ELEVATOR, k])
    colors = ["#4b3f8c" if v >= 0 else "#e07060" for v in shape]
    ax.bar(stations, shape, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Measurement Station")
    ax.set_ylabel("Normalized Amplitude")
    ax.set_title(f"Flexible Mode  (f = {f_mode:.3f} Hz, zeta = {z_mode:.2f}%)")
    ax.set_ylim([-1.2, 1.2])
    ax.grid(axis="y", alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_mode_shapes.png"), **STYLE)
plt.close(fig)
print("Saved fig2_mode_shapes.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2b -- Bode plot: Z acc POINT B / Elevators (matches notebook)
# ─────────────────────────────────────────────────────────────────────────────
out_b  = 1   # Z acc POINT B (0-based)
in_elv = IN_ELEVATOR
H_bode = TF[out_b, in_elv, :]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(freq, np.abs(H_bode), color="#e8a020", lw=1.8)
ax1.set_ylabel("Gain")
ax1.set_title("TF: Z acc POINT B / Elevators")
ax1.grid(True, alpha=0.35)
for f_m in [f1_flex, f2_flex]:
    ax1.axvline(f_m, color="tab:gray", lw=0.8, ls="--", alpha=0.6)

ax2.plot(freq, np.angle(H_bode, deg=True), color="#e8a020", lw=1.8)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Phase (deg)")
ax2.set_ylim([-185, 185])
ax2.set_yticks([-150, -100, -50, 0, 50, 100, 150])
ax2.grid(True, alpha=0.35)
for f_m in [f1_flex, f2_flex]:
    ax2.axvline(f_m, color="tab:gray", lw=0.8, ls="--", alpha=0.6)

ax2.set_xlim([0, 5])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2b_bode_accB_elevator.png"), **STYLE)
plt.close(fig)
print("Saved fig2b_bode_accB_elevator.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3b -- Output 11 identification (comparison with acc E and acc F)
# ─────────────────────────────────────────────────────────────────────────────
# TF magnitudes from wind input -- used to identify what output 11 measures
H_accE = np.abs(TF[7,  IN_WIND, :])   # acc E
H_accF = np.abs(TF[8,  IN_WIND, :])   # acc F
H_out11= np.abs(TF[10, IN_WIND, :])   # output 11 (sensor)
H_diff = np.abs(TF[8,  IN_WIND, :] - TF[7, IN_WIND, :])  # acc F - acc E

# rigid-body and flex-1 mode frequencies
f_rigid = freq_hz[(freq_hz > 0.01) & (np.abs(zeta) < 95) & (freq_hz < 0.5)][0]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(freq, H_accE,  color="tab:blue",   lw=1.6, label="acc E  (output 8)")
ax.plot(freq, H_accF,  color="tab:orange", lw=1.6, label="acc F  (output 9)")
ax.plot(freq, H_out11, color="tab:red",    lw=1.8, label="output 11  (?)")
ax.plot(freq, H_diff,  color="k",          lw=1.5, ls="--", label="acc F - acc E")
ax.axvline(f_rigid, color="k", ls=":",  lw=1.0)
ax.axvline(f1_flex, color="k", ls=":",  lw=1.0)
ax.text(f_rigid + 0.03, ax.get_ylim()[1] * 0.92, "rigid-body",
        fontsize=8, va="top")
ax.text(f1_flex + 0.03, ax.get_ylim()[1] * 0.92, "flex mode 1",
        fontsize=8, va="top")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Gain")
ax.set_title("Identifying Output 11 -- comparison with acc E and acc F")
ax.set_xlim([0, 5])
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3b_output11_identification.png"), **STYLE)
plt.close(fig)
print("Saved fig3b_output11_identification.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Von Karman PSD
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(freq, phi_vk, color="tab:blue", lw=1.8)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD  [(m/s)$^2$/(rad/s)]")
ax.set_title("Von Karman Turbulence Spectrum  ($\\sigma$=21.58 m/s,  L=762 m)")
ax.grid(True, which="both", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_von_karman_psd.png"), **STYLE)
plt.close(fig)
print("Saved fig3_von_karman_psd.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Wing-root bending moment PSD (open-loop)
# ─────────────────────────────────────────────────────────────────────────────
S_mx = S_y[OUT_MX_ROOT, :]
rms_ol = rms_from_psd(S_mx, omega)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(freq, S_mx * 1e-9, color="tab:blue", lw=1.8)
ax.axvline(f1_flex, color="tab:red", ls="--", lw=1.2, label=f"Flex 1  ({f1_flex:.3f} Hz)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD  [mDaN$^2$/(rad/s)]  $\\times 10^9$")
ax.set_title(f"Wing-Root Bending Moment PSD — Open-loop\n"
             f"RMS Mx = {rms_ol:.3e} mDaN")
ax.legend(); ax.grid(True, alpha=0.4)
ax.set_xlim([0, 5])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_mx_psd_openloop.png"), **STYLE)
plt.close(fig)
print("Saved fig4_mx_psd_openloop.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 -- Nyquist 2x2 MIMO grid (empty law, Gain=2, Filter=1.2 Hz)
# Each panel: L_ij = H_plant[sensor, aileron_j] * H_law  (positive feedback)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.signal import butter as _butter
b_e, a_e = _butter(2, 2 * np.pi * 1.2, btype="low", analog=True)
H_law_g2    = law_freq_response(2.0, b_e.tolist(), a_e.tolist(), TAU, omega)
H_inner_in  = TF[OUT_SENSOR, IN_INNER_AILERON, :] * H_law_g2
H_outer_in  = TF[OUT_SENSOR, IN_OUTER_AILERON, :] * H_law_g2

# 2x2 layout: rows = output channel (inner/outer), cols = input channel (inner/outer)
# H_ol[out_i, in_j] = H_plant[sensor, in_j] * H_law  (same law for both outputs)
panels = [
    (H_inner_in, "In: Inner Ail", "Out: Inner Ail"),
    (H_outer_in, "In: Outer Ail", "Out: Inner Ail"),
    (H_inner_in, "In: Inner Ail", "Out: Outer Ail"),
    (H_outer_in, "In: Outer Ail", "Out: Outer Ail"),
]
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
fig.suptitle("Nyquist -- Empty Law (Gain=2, Filter=1.2)", fontsize=12)
for ax, (H_ol, col_lbl, row_lbl) in zip(axes.flat, panels):
    ax.plot(np.real(H_ol),  np.imag(H_ol),  color="#1a1a6e", lw=1.4)
    ax.plot(np.real(H_ol), -np.imag(H_ol),  color="#1a1a6e", lw=1.4, alpha=0.4)
    ax.plot(1, 0, "rx", ms=10, mew=2)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Real"); ax.set_ylabel("Imag")
    ax.set_title(f"{col_lbl}\n{row_lbl}", fontsize=8)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_nyquist_empty_law.png"), **STYLE)
plt.close(fig)
print("Saved fig5_nyquist_empty_law.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Open-loop vs Closed-loop PSD comparison
# ─────────────────────────────────────────────────────────────────────────────
b_d, a_d = _butter(LAW_FILTER_ORDER, 2 * np.pi * LAW_CUTOFF_HZ, btype="low", analog=True)
A_cl, B_cl, C_cl, D_cl = build_closed_loop(
    A, B, C, D,
    Gain=LAW_GAIN,
    num_c=b_d.tolist(), den_c=a_d.tolist(), tau=TAU,
    sensor_idx=OUT_SENSOR,
    cs_indices=[IN_INNER_AILERON, IN_OUTER_AILERON],
    ext_input_slice=slice(IN_ELEVATOR, None)   # inputs 2,3 (elevators, wind)
)
TF_CL   = compute_tf(A_cl, B_cl, C_cl, D_cl, omega)
# wind is now input index 1 in the CL model (0=elevators, 1=wind)
S_y_cl  = output_psd(TF_CL, phi_vk, 1, VTAS)
S_mx_cl = S_y_cl[OUT_MX_ROOT, :]
rms_cl  = rms_from_psd(S_mx_cl, omega)
red_pct = (rms_ol - rms_cl) / rms_ol * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(freq, S_mx * 1e-9,    color="tab:blue",   lw=2,   label=f"No law  RMS={rms_ol:.2e} mDaN")
ax.plot(freq, S_mx_cl * 1e-9, color="tab:red",    lw=2,   label=f"TLA law  RMS={rms_cl:.2e} mDaN")
ax.axvline(f1_flex, color="k", ls=":", lw=1.2, label=f"Flex 1  ({f1_flex:.3f} Hz)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD  [mDaN$^2$/(rad/s)]  $\\times 10^9$")
ax.set_title(f"PSD Wing Root Bending Moment: Open-loop vs Closed-loop\n"
             f"RMS reduction = {red_pct:.1f}%")
ax.set_xlim([0, 5])
ax.legend(); ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_psd_comparison.png"), **STYLE)
plt.close(fig)
print("Saved fig6_psd_comparison.png")

print("\nAll figures saved to:", FIG_DIR)
