# TP3 — Turbulence Response and Load Alleviation of a Flexible Aircraft

**ISAE-SUPAERO** | Master in Aerospace Engineering  
**Author:** Phani Raghava PANCHAGNULA

---

This project analyses the aeroelastic behaviour of a large flexible aircraft (A340-class)
at cruise, covers its structural response to atmospheric turbulence, and implements an
active control law to reduce wing loads. The analysis is done in Python using a
linear state-space model.

For the full write-up see **[REPORT.md](REPORT.md)**.

---

## Quick start

```bash
python save_figures.py   # regenerate all PNG figures
pytest tests/ -v         # run unit tests (25 tests)
jupyter notebook code.ipynb
```

Use the `gnn_surrogate` conda environment. In VS Code, set the interpreter to that
environment (`Ctrl+Shift+P` → `Python: Select Interpreter`) to clear scipy import warnings.

---

## Project layout

```
Turbulence_and_Flexible_Aircraft/
├── REPORT.md                   <- full detailed report
├── code.ipynb                  <- main notebook
├── save_figures.py             <- regenerate all figures
├── src/
│   ├── config.py               <- constants and I/O indices
│   ├── modal.py                <- eigenvalue analysis
│   ├── transfer_functions.py   <- H(jw) = C(jwI-A)^-1 B + D
│   ├── turbulence.py           <- Von Karman PSD
│   ├── psd_response.py         <- output PSD and RMS
│   └── control_law.py          <- TLA law, margins, closed-loop
├── tests/                      <- pytest (25 tests, all passing)
├── models/AC_model.mat         <- state-space model (read-only)
├── results/figures/            <- saved PNG figures
```

## Key results

| | Value |
|--|-------|
| First flexible mode | 1.193 Hz, 10.1% damping |
| RMS Mx open-loop | 1.119 x 10^6 mDaN |
| RMS Mx closed-loop | 1.032 x 10^6 mDaN |
| RMS reduction | **7.8%** |
| Control law gain | K = −4.3 (negative, positive feedback) |
| Gain margin | 10.76 |
