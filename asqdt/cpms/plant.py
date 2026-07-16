"""
CPMS Plant, Observer-Form Digital Twin, and Attack Generators.
State:   x = [p1,p2,p3, v1,v2,v3]
Input:   u = [τ1,τ2,τ3]
Output:  y = Cx + C_a·a + v
Quality: z = x (full state, W_z=I_6 for CNC)
"""
import numpy as np
from scipy.linalg import expm
from .. import config as cfg


def _build_system(mismatch_rng=None, mismatch_frac=0.0):
    masses  = np.array([2.0, 1.5, 1.0])
    damping = np.array([0.8, 0.6, 0.4])
    if mismatch_rng is not None and mismatch_frac > 0:
        masses  = masses  * (1 + mismatch_frac * mismatch_rng.randn(3))
        damping = damping * (1 + mismatch_frac * mismatch_rng.randn(3))

    n, m = cfg.N_STATE, cfg.N_INPUT
    Ac = np.zeros((n, n))
    Bc = np.zeros((n, m))
    for i in range(3):
        Ac[i, i+3] = 1.0
        Ac[i+3, i+3] = -damping[i] / masses[i]
        Bc[i+3, i] = 1.0 / masses[i]
    dt = cfg.DT_STEP
    A = expm(Ac * dt)
    B = dt * Bc

    C = np.eye(n)
    B_a = np.zeros((n, cfg.N_ATTACK))
    B_a[:3, :3] = 0.1 * np.eye(3)
    B_a[3:, 3:] = 0.05 * np.eye(3)
    C_a = np.zeros((n, cfg.N_ATTACK))
    C_a[:3, :3] = 0.1 * np.eye(3)
    return A, B, B_a, C, C_a


class CPMSPlant:
    """Physical CPMS (ground truth)."""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.A, self.B, self.B_a, self.C, self.C_a = _build_system()
        self.x = np.zeros(cfg.N_STATE)

    def reset(self, x0=None):
        self.x = x0 if x0 is not None else 0.1 * self.rng.randn(cfg.N_STATE)
        return self.x.copy()

    def step(self, u, a=None):
        if a is None:
            a = np.zeros(cfg.N_ATTACK)
        # FIX: truncated Gaussian noise — clip component-wise (paper Section VII-B)
        w = np.clip(cfg.NOISE_W * self.rng.randn(cfg.N_STATE),
                     -cfg.NOISE_W, cfg.NOISE_W)
        v = np.clip(cfg.NOISE_V * self.rng.randn(cfg.N_OUTPUT),
                     -cfg.NOISE_V, cfg.NOISE_V)
        self.x = self.A @ self.x + self.B @ u + self.B_a @ a + w
        y = self.C @ self.x + self.C_a @ a + v
        return self.x.copy(), y

    def reference(self, H):
        ref = np.zeros((H, cfg.N_QUALITY))
        t_sw = 20 + self.rng.randint(0, 10)
        target = 0.3 * (self.rng.rand(cfg.N_QUALITY) - 0.5)
        ref[t_sw:] = target
        return ref

    @staticmethod
    def pd_controller(x, p_ref):
        Kp, Kd = 10.0, 2.0
        u = Kp * (p_ref - x[:3]) + Kd * (-x[3:])
        return np.clip(u, -5, 5)


class ObserverDT:
    """Observer-form digital twin (Eq. 4-5)."""

    def __init__(self, mismatch_frac=0.0, seed=None):
        rng = np.random.RandomState(seed)
        self.A, self.B, _, self.C, _ = _build_system(rng, mismatch_frac)
        self.K_o = cfg.OBSERVER_GAIN * np.eye(cfg.N_STATE, cfg.N_OUTPUT)
        self.x_hat = np.zeros(cfg.N_STATE)

    def reset(self, x0=None):
        self.x_hat = x0 if x0 is not None else np.zeros(cfg.N_STATE)

    def step(self, u, y, sensor_mask=None):
        y_hat = self.C @ self.x_hat
        innovation = y - y_hat
        if sensor_mask is not None:
            innovation = sensor_mask * innovation
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.K_o @ innovation
        return y_hat, self.x_hat.copy()

    def residual(self, y, y_hat, sensor_mask):
        return sensor_mask * (y - y_hat)


# ── Legacy detectors (per-benchmark, paper Section III-B) ─────────

def chi_squared_legacy(residuals, Sigma_r_inv=None):
    """
    Chi-squared legacy detector for CNC and Tank.
    D_leg(a) = max_k r_k^T Σ_r^{-1} r_k
    If Sigma_r_inv is None, estimate from residuals (identity fallback).
    """
    if Sigma_r_inv is None:
        # Estimate covariance from residuals and invert
        cov = np.cov(residuals.T) + 1e-6 * np.eye(residuals.shape[1])
        try:
            Sigma_r_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            Sigma_r_inv = np.eye(residuals.shape[1])
    scores = np.array([r @ Sigma_r_inv @ r for r in residuals])
    return np.max(scores)


def cusum_legacy(residuals, mu_0=None):
    """
    CUSUM legacy detector for CSTR.
    D_leg(a) = max_k S_k^+ where S_k^+ = max(0, S_{k-1}^+ + ||r_k|| - μ_0)
    If mu_0 is None, estimate from first 20% of residuals.
    """
    norms = np.linalg.norm(residuals, axis=1)
    if mu_0 is None:
        n_cal = max(1, len(norms) // 5)
        mu_0 = np.mean(norms[:n_cal]) + 2 * np.std(norms[:n_cal])
    S = 0.0
    S_max = 0.0
    for n in norms:
        S = max(0.0, S + n - mu_0)
        S_max = max(S_max, S)
    return S_max


def legacy_detector_response(residuals, method='chi_sq', **kwargs):
    """Dispatch to the correct legacy detector."""
    if method == 'chi_sq':
        return chi_squared_legacy(residuals, kwargs.get('Sigma_r_inv'))
    elif method == 'cusum':
        return cusum_legacy(residuals, kwargs.get('mu_0'))
    else:
        raise ValueError(f"Unknown legacy detector: {method}")


# ── Impact computation (paper Section III-B) ──────────────────────

def compute_impact(z_traj, z_nom, W_z=None):
    """
    Impact I(a) = Σ_k ||z_k - z_k^0||_{W_z}^2
    W_z: weight matrix. None defaults to identity.
    """
    diff = z_traj - z_nom
    if W_z is None:
        return np.sum(diff ** 2)
    else:
        return sum(d @ W_z @ d for d in diff)


# ── Attack generators (CNC, E_max=50) ────────────────────────────

def _normalize_energy(a_seq, E_max):
    energy = np.sum(a_seq ** 2)
    if energy > E_max:
        a_seq *= np.sqrt(E_max / energy)
    return a_seq


def generate_fdi_attack(H, rng, E_max=2.0):
    a = np.zeros((H, cfg.N_ATTACK))
    bias = 0.8 * rng.randn(3)
    t_start = 30 + rng.randint(0, 20)
    a[t_start:, :3] = bias
    return _normalize_energy(a, E_max)


def generate_command_attack(H, rng, E_max=2.0):
    a = np.zeros((H, cfg.N_ATTACK))
    freq = 0.5 + rng.rand()
    t_start = 25 + rng.randint(0, 15)
    t = np.arange(H)
    amp = 0.6 * rng.rand(3) + 0.3
    for i in range(3):
        phase = rng.rand() * 2 * np.pi
        a[t_start:, 3+i] = amp[i] * np.sin(2*np.pi*freq*t[t_start:]*cfg.DT_STEP + phase)
    return _normalize_energy(a, E_max)


def generate_ramp_attack(H, rng, E_max=2.0):
    a = np.zeros((H, cfg.N_ATTACK))
    t_start = 25 + rng.randint(0, 15)
    direction = rng.randn(3)
    direction /= np.linalg.norm(direction) + 1e-8
    ramp_len = H - t_start
    for k in range(ramp_len):
        a[t_start + k, :3] = direction * (k / ramp_len) * 0.8
    return _normalize_energy(a, E_max)


def generate_composite_attack(H, rng, E_max=2.0):
    a1 = generate_fdi_attack(H, rng, E_max * 2)
    a2 = generate_command_attack(H, rng, E_max * 2)
    a = 0.6 * a1 + 0.4 * a2
    return _normalize_energy(a, E_max)


ATTACK_GENERATORS = {
    'fdi': generate_fdi_attack,
    'command': generate_command_attack,
    'ramp': generate_ramp_attack,
    'composite': generate_composite_attack,
}
