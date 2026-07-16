"""
AS-QDT Configuration — all hyperparameters in one place for reproducibility.
Aligned with paper Section VII (Experimental Evaluation).
"""

# ── System (3-axis CNC-like CPMS) ──────────────────────────────────
N_STATE      = 6        # [p1,p2,p3, v1,v2,v3]
N_INPUT      = 3        # torques
N_OUTPUT     = 6        # full-state measurement
N_ATTACK     = 6        # attack channels
N_QUALITY    = 3        # quality = position tracking error
HORIZON      = 120      # timesteps per episode
DT_STEP      = 0.01     # discretization (s)
NOISE_W      = 0.005    # process noise bound (||w_k||_∞ ≤ NOISE_W)
NOISE_V      = 0.01     # measurement noise bound (||v_k||_∞ ≤ NOISE_V)

# ── Observer-form DT ───────────────────────────────────────────────
OBSERVER_GAIN = 0.3

# ── Twin mismatch ─────────────────────────────────────────────────
MISMATCH_FRAC = 0.05

# ── Per-benchmark attack-set parameters ───────────────────────────
# Calibrated for correct legacy detectors (chi-sq for CNC/Tank, CUSUM for CSTR)
ATTACK_SET = {
    'CNC':  dict(rho=0.5, epsilon=100.0,  E_max=2.0,   W_L=10),
    'CSTR': dict(rho=0.2, epsilon=300.0,  E_max=500.0, W_L=8),
    'Tank': dict(rho=0.1, epsilon=50.0,   E_max=20.0,  W_L=8),
}
# Legacy defaults (CNC)
RHO     = 0.5
EPSILON = 100.0
E_MAX   = 2.0

# ── Windowed features ──────────────────────────────────────────────
WINDOW_LEN  = 10   # W_L for CNC; CSTR/Tank override to 8
FEATURE_DIM = N_OUTPUT * WINDOW_LEN  # d = 60

# ── Quantum kernel ─────────────────────────────────────────────────
N_QUBITS      = 4
CIRCUIT_DEPTH = 3
N_SHOTS       = 0   # exact statevector

# ── Surrogates ─────────────────────────────────────────────────────
NYSTROM_RANK      = 10
LEARNED_HIDDEN    = (64, 32, 16)
LEARNED_MAX_ITER  = 500
LEARNED_EMBED_DIM = 15

# ── Sensor selection ───────────────────────────────────────────────
SENSOR_BUDGET = 4

# ── Detection ──────────────────────────────────────────────────────
DELTA         = 0.05
N_PERM_THRESH = 150   # permutation threshold
N_PERM_SIG    = 500   # significance test

# ── Co-design solver ──────────────────────────────────────────────
MAX_ITERS   = 6
LAMBDA_Q    = 0.0
KERNEL_LR   = 0.05

# ── Data generation ───────────────────────────────────────────────
N_EPISODES_NOM = 20
N_EPISODES_ATK = 10
SEED  = 42
SEEDS = [42, 123, 256, 789, 1024]
