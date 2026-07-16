"""
Coupled Two-Tank System (Nonlinear).
Quality: z = [h1, h2], W_z = I_2
Legacy detector: chi-squared on linearized observer residuals
"""
import numpy as np
from ..cpms.plant import chi_squared_legacy, compute_impact


class CoupledTankPlant:
    def __init__(self, seed=None, mismatch_frac=0.0):
        self.rng = np.random.RandomState(seed)
        self.A1 = 28.0; self.A2 = 32.0
        self.a1 = 0.071 * (1 + mismatch_frac * self.rng.randn())
        self.a2 = 0.057 * (1 + mismatch_frac * self.rng.randn())
        self.a3 = 0.030 * (1 + mismatch_frac * self.rng.randn())
        self.dt = 1.0
        self.n_state = 2; self.n_input = 1; self.n_output = 2; self.n_attack = 2
        self.noise_v = 0.05  # measurement noise scale
        self.x = np.array([15.0, 15.0])

    def _dynamics(self, x, u, a=None):
        h1, h2 = np.clip(x, 0.1, 50.0)
        q_in = np.clip(u[0], 0, 10)
        if a is None: a = np.zeros(2)
        dh1 = (q_in + self.a3*np.sqrt(h2) - self.a1*np.sqrt(h1) + 0.5*a[0]) / self.A1
        dh2 = (self.a1*np.sqrt(h1) - self.a2*np.sqrt(h2) + 0.3*a[1]) / self.A2
        return np.array([dh1, dh2])

    def reset(self, x0=None):
        if x0 is not None: self.x = x0.copy()
        else: self.x = np.array([15.0 + self.rng.randn(), 15.0 + self.rng.randn()])
        return self.x.copy()

    def step(self, u, a=None):
        k1 = self._dynamics(self.x, u, a)
        k2 = self._dynamics(self.x + 0.5*self.dt*k1, u, a)
        k3 = self._dynamics(self.x + 0.5*self.dt*k2, u, a)
        k4 = self._dynamics(self.x + self.dt*k3, u, a)
        self.x = self.x + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.x = np.clip(self.x, 0.1, 50.0)
        # FIX: truncated Gaussian noise
        v_raw = self.noise_v * self.rng.randn(self.n_output)
        v = np.clip(v_raw, -self.noise_v, self.noise_v)
        y = self.x + v
        return self.x.copy(), y

    def reference(self, H):
        ref = np.zeros((H, 1))
        t_sw = 15 + self.rng.randint(0, 10)
        target = 15.0 + 3.0 * self.rng.randn()
        ref[t_sw:] = target
        return ref

    @staticmethod
    def pi_controller(x, h2_ref, integral_err=0.0):
        err = h2_ref - x[1]
        u = np.array([3.0 + 2.0*err + 0.1*integral_err])
        return np.clip(u, 0, 10), err


class TankObserverDT:
    def __init__(self, mismatch_frac=0.0, seed=None):
        self.model = CoupledTankPlant(seed=seed, mismatch_frac=mismatch_frac)
        self.K_o = 0.2 * np.eye(2)
        self.x_hat = np.array([15.0, 15.0])

    def reset(self, x0=None):
        self.x_hat = x0.copy() if x0 is not None else np.array([15.0, 15.0])

    def step(self, u, y, sensor_mask=None):
        y_hat = self.x_hat.copy()
        inn = y - y_hat
        if sensor_mask is not None: inn = sensor_mask[:2] * inn
        k1 = self.model._dynamics(self.x_hat, u)
        k2 = self.model._dynamics(self.x_hat + 0.5*self.model.dt*k1, u)
        k3 = self.model._dynamics(self.x_hat + 0.5*self.model.dt*k2, u)
        k4 = self.model._dynamics(self.x_hat + self.model.dt*k3, u)
        self.x_hat = self.x_hat + (self.model.dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.x_hat += self.K_o @ inn
        self.x_hat = np.clip(self.x_hat, 0.1, 50.0)
        return y_hat, self.x_hat.copy()

    def residual(self, y, y_hat, mask):
        return mask[:2] * (y - y_hat)


# ── Attack generators (E_max=20, paper Section VII-B) ─────────────

def _ne(a, E_max):
    e = np.sum(a**2)
    if e > E_max: a *= np.sqrt(E_max / e)
    return a

def tank_bias_attack(H, rng, E_max=20.0):
    a = np.zeros((H, 2))
    t_start = 15 + rng.randint(0, 10)
    a[t_start:, 0] = 0.4 * rng.randn()
    return _ne(a, E_max)

def tank_oscillation_attack(H, rng, E_max=20.0):
    a = np.zeros((H, 2))
    t_start = 15 + rng.randint(0, 10)
    freq = 0.2 + 0.3*rng.rand()
    t = np.arange(H)
    a[t_start:, 0] = 0.3 * np.sin(2*np.pi*freq*t[t_start:])
    a[t_start:, 1] = 0.2 * np.cos(2*np.pi*freq*t[t_start:])
    return _ne(a, E_max)

def tank_drift_attack(H, rng, E_max=20.0):
    a = np.zeros((H, 2))
    t_start = 15 + rng.randint(0, 10)
    ramp_len = H - t_start
    for k in range(ramp_len):
        a[t_start+k, 1] = 0.4 * (k/ramp_len)
    return _ne(a, E_max)

TANK_ATTACK_GENERATORS = {
    'tank_bias': tank_bias_attack,
    'tank_oscillation': tank_oscillation_attack,
    'tank_drift': tank_drift_attack,
}


def run_tank_episode(plant, twin, ref, attack_seq=None, sensor_mask=None):
    H = len(ref)
    if sensor_mask is None: sensor_mask = np.ones(2)
    x0 = plant.reset(); twin.reset(x0.copy())
    residuals = np.zeros((H, 2))
    # FIX: quality z_k = [h1, h2] (2-vector)
    z_traj = np.zeros((H, 2))
    ie = 0.0
    for k in range(H):
        h2r = ref[k, 0] if ref[k, 0] != 0 else 15.0
        u, err = CoupledTankPlant.pi_controller(plant.x, h2r, ie)
        ie += err * plant.dt
        a = attack_seq[k] if attack_seq is not None else None
        x, y = plant.step(u, a)
        yh, xh = twin.step(u, y, sensor_mask)
        residuals[k] = twin.residual(y, yh, sensor_mask)
        z_traj[k] = x   # quality = [h1, h2]
    return {'residuals': residuals, 'z_traj': z_traj}


def generate_tank_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10, H=120, seed=42):
    """Generate Tank dataset: one window per episode, chi-squared legacy."""
    W_L = 8; mask = np.ones(2)

    # Nominal
    all_nom_residuals = []; nom_windows = []
    for ep in range(n_nom):
        p = CoupledTankPlant(seed=seed+100+ep)
        t = TankObserverDT(mismatch_frac=mismatch_frac, seed=seed+1)
        ref = p.reference(H)
        d = run_tank_episode(p, t, ref, sensor_mask=mask)
        all_nom_residuals.append(d['residuals'])
        k0 = H//2 - W_L//2
        nom_windows.append(d['residuals'][k0:k0+W_L].flatten())

    nom = np.nan_to_num(np.array(nom_windows))
    mu = nom.mean(0); std = nom.std(0) + 1e-8
    nom = (nom - mu) / std

    # Estimate Σ_r^{-1} for chi-squared legacy
    all_res = np.vstack(all_nom_residuals)
    cov = np.cov(all_res.T) + 1e-6 * np.eye(2)
    try: Sig_inv = np.linalg.inv(cov)
    except: Sig_inv = np.eye(2)

    # Attacks
    atk_data = {}
    for aname, agen in TANK_ATTACK_GENERATORS.items():
        aw = []; impacts = []; leg_resps = []; energies = []
        for ep in range(n_atk):
            p = CoupledTankPlant(seed=seed+200+ep)
            t = TankObserverDT(mismatch_frac=mismatch_frac, seed=seed+1)
            ref = p.reference(H)
            a_seq = agen(H, np.random.RandomState(seed+300+ep), E_max=20.0)

            # Nominal for impact
            p_n = CoupledTankPlant(seed=seed+200+ep)
            t_n = TankObserverDT(mismatch_frac=mismatch_frac, seed=seed+1)
            d_n = run_tank_episode(p_n, t_n, ref, sensor_mask=mask)

            d = run_tank_episode(p, t, ref, a_seq, mask)
            impact = compute_impact(d['z_traj'], d_n['z_traj'])
            leg = chi_squared_legacy(d['residuals'], Sig_inv)
            energy = np.sum(a_seq**2)

            k0 = H//2 - W_L//2
            aw.append(d['residuals'][k0:k0+W_L].flatten())
            impacts.append(impact); leg_resps.append(leg); energies.append(energy)

        aw_arr = np.nan_to_num(np.array(aw))
        atk_data[aname] = {
            'windows': (aw_arr - mu) / std, 'impacts': impacts,
            'legacy_responses': leg_resps, 'energies': energies, 'label': aname,
        }
    return nom, atk_data
