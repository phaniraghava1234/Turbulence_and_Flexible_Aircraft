# PROGRESS.md — TP3 Flexible Aircraft

**Rule:** Keep this file under 200 lines at all times.
When it reaches ~180 lines: move completed sessions older than 2 sessions
into `PROGRESS_archive.md`, keep only the last 2 + current state + next task.

---

## Current State

- **Last completed section:** Part 3 — Control Law (Q9 + Q10 answered, CL eigenvalue table done)
- **Last passing test:** — (src/ modules not yet extracted)
- **Notebook:** `code.ipynb` — all 4 parts complete through Q10
- **Open issues / blockers:** none

## Next Session Starts At

- **Section:** Wrap-up — extract notebook functions to `src/` modules + write pytest tests
- **First task:** Create `src/config.py`, `src/modal.py`, `src/turbulence.py`, `src/transfer_functions.py`, `src/psd_response.py`, `src/control_law.py` and corresponding `tests/` files. Ask user for permission to write files first.

---

## Session Log

## Session 2026-05-08

### Completed
- [x] Part 0: Loaded AC_model.mat, verified A(26×26) B(26×4) C(11×26) D(11×4), extracted labels
- [x] Part 1: Modal analysis — freq_damp(), eigenvalue table, mode scatter plot, mode shape plot
- [x] Part 2: Von Karman PSD, TF computation, wing root bending moment PSD, RMS=1.119e6 mDaN
- [x] Part 3: Output 11 identified (acc_F − acc_E), Nyquist empty law (GM=2.48), interactive design tool
- [x] Part 3: Finalized design: Gain=-4.3, lowpass order=2, cutoff=1.20 Hz, GM=10.76 (positive feedback)
- [x] Part 3: Closed-loop built (29 states), PSD comparison: RMS -7.8% (1.119e6 → 1.032e6 mDaN)
- [x] Part 3: CL eigenvalue table — flex 1 shifts 1.193→1.350 Hz (+13%), damping barely changes
- [x] Q9 markdown: law structure, parameters, margins
- [x] Q10 markdown: CL eigenvalue table + frequency-detuning interpretation

### Key results
- VTAS = 247.6 m/s, L = 762 m, Usigma = 21.582 m/s
- Flex mode 1: 1.193 Hz, zeta=10.1% (OL); 1.350 Hz, zeta=9.4% (CL)
- RMS Mx: 1.119e6 mDaN (OL) → 1.032e6 mDaN (CL), -7.8%
- Negative gain required (positive feedback, Nyquist lobe must stay left of +1)

### State at end
- Last notebook section: Cell 4.9 + Q9/Q10 markdown
- Last passing test: none yet (no src/ files)
- Open issues: none
