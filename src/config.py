"""
config.py -- Flight point constants and model I/O index definitions.

All physical constants for the TP3 study case are defined here.
No other module should hardcode these values.

Flight point: Mach 0.8, altitude 25000 ft (cruise).
"""

# -- Flight point --------------------------------------------------------------
MACH        = 0.8          # Mach number (-)
ALTITUDE_FT = 25_000       # Pressure altitude (ft)
VTAS        = 247.6        # True airspeed (m/s)

# -- Turbulence (Von Karman, MIL-SPEC moderate intensity) ----------------------
L      = 762.0             # Turbulence scale length (m)
USIGMA = 21.5819           # Turbulence intensity -- std dev of vertical gust (m/s)

# -- Actuator ------------------------------------------------------------------
TAU = 0.1                  # First-order actuator time constant (s)

# -- Input indices (0-based) ---------------------------------------------------
IN_INNER_AILERON = 0       # Inner ailerons command (rad)
IN_OUTER_AILERON = 1       # Outer ailerons command (rad)
IN_ELEVATOR      = 2       # Elevator command (rad)
IN_WIND          = 3       # Vertical wind incidence (rad) -- NOT wind speed

# -- Output indices (0-based) --------------------------------------------------
OUT_ACC_A      = 0         # Z acceleration at point A (m/s^2)
OUT_ACC_B      = 1         # Z acceleration at point B (m/s^2)
OUT_ACC_C      = 2         # Z acceleration at point C (m/s^2)
OUT_ACC_D      = 3         # Z acceleration at point D (m/s^2)
OUT_PITCH_RATE = 4         # Pitch rate (rad/s)
OUT_Z_SPEED_G  = 5         # Z speed at centre of gravity G (m/s)
OUT_PITCH_ANGLE= 6         # Pitch angle at G (rad)
OUT_ACC_E      = 7         # Z acceleration at point E (m/s^2)
OUT_ACC_F      = 8         # Z acceleration at point F (m/s^2)
OUT_MX_ROOT    = 9         # Wing-root bending moment Mx (mDaN)
OUT_SENSOR     = 10        # TLA sensor: acc_F - acc_E (m/s^2)

# -- Control law design (finalised) --------------------------------------------
LAW_GAIN        = -4.3     # Negative: lobe left of +1 in positive-feedback Nyquist
LAW_FILTER_TYPE = "lowpass"
LAW_FILTER_ORDER= 2
LAW_CUTOFF_HZ   = 1.20     # Butterworth cutoff (Hz), just above flex-mode 1
