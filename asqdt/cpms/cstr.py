"""
Nonlinear CSTR Benchmark.
State: x = [C_A, C_B, T, T_c]
Quality: z = [C_A, C_B, T] with W_z = diag(1, 1, 0.01)
Legacy detector: CUSUM on residual magnitude
"""
import numpy as np
from ..cpms.plant import cusum_legacy, compute_impact


# Quality weight matrix (paper Section VII-B)
W_Z_CSTR = np.diag([1.0, 1.0, 0.01])


class CSTRPlant:
    def __init__(self, seed=None, mismatch_frac=0.0):
        self.rng = np.random.RandomState(seed)
        self.V = 100.0; self.rho = 1000.0; self.Cp = 0.239
        self.dH = -5e4; self.E_R = 8750.0; self.k0 = 7.2e10
        self.UA = 5e4; self.CA_in = 1.0; self.T_in = 350.0
        if mismatch_frac > 0:
            self.k0 *= (1 + mismatch_frac * self.rng.randn())
            self.E_R *= (1 + mismatch_frac * self.rng.randn())
            self.UA *= (1 + mismatch_frac * self.rng.randn())
        self.n_state = 4; self.n_input = 2; self.n_output = 3; self.n_attack = 3
        self.dt = 0.1
        # Noise scales per output channel
        self.noise_v = np.array([0.005, 0.005, 0.5])
        self.x = np.array([0.5, 0.5, 350.0, 350.0])

    def _reaction_rate(self, CA, T):
        T_safe = np.clip(T, 250.0, 500.0)
        CA_safe = np.clip(CA, 0.0, 5.0)
        return self.k0 * np.exp(-self.E_R / T_safe) * CA_safe

    def _dynamics(self, x, u, a=None):
        CA, CB, T, Tc = x
        F, Q = u
        if a is None: a = np.zeros(self.n_attack)
        F = max(F, 1.0); tau = self.V / F
        r = self._reaction_rate(CA + 0.1 * a[0], T)
        dCA = (self.CA_in - CA) / tau - r
        dCB = -CB / tau + r
        dT = ((self.T_in - T) / tau + (-self.dH / (self.rho * self.Cp)) * r
              - self.UA / (self.V * self.rho * self.Cp) * (T - Tc) + 0.05 * a[1])
        dTc = (Q + self.UA * (T - Tc)) / (50.0 * self.rho * self.Cp) + 0.02 * a[2]
        return np.array([dCA, dCB, dT, dTc])

    def reset(self, x0=None):
        if x0 is not None:
            self.x = x0.copy()
        else:
            self.x = np.array([0.5 + 0.05*self.rng.randn(), 0.5 + 0.05*self.rng.randn(),
                               350.0 + 2.0*self.rng.randn(), 350.0 + 1.0*self.rng.randn()])
        return self.x.copy()

    def step(self, u, a=None):
        dt = self.dt
        k1 = self._dynamics(self.x, u, a)
        k2 = self._dynamics(self.x + 0.5*dt*k1, u, a)
        k3 = self._dynamics(self.x + 0.5*dt*k2, u, a)
        k4 = self._dynamics(self.x + dt*k3, u, a)
        self.x = self.x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        self.x[0] = np.clip(self.x[0], 0.0, 3.0)
        self.x[1] = np.clip(self.x[1], 0.0, 3.0)
        self.x[2] = np.clip(self.x[2], 280.0, 450.0)
        self.x[3] = np.clip(self.x[3], 280.0, 420.0)
        # FIX: truncated Gaussian noise — clip component-wise
        v_raw = self.noise_v * self.rng.randn(self.n_output)
        v = np.clip(v_raw, -self.noise_v, self.noise_v)
        y = self.x[:3] + v
        return self.x.copy(), y

    def reference(self, H):
        ref = np.zeros((H, 1))
        t_sw = 15 + self.rng.randint(0, 10)
        target = 0.5 + 0.15 * self.rng.randn()
        ref[t_sw:] = target
        return ref

    @staticmethod
    def pi_controller(x, CB_ref, integral_err=0.0):
        CB = x[1]; err = CB_ref - CB
        F = 100.0 + 50.0*err + 5.0*integral_err
        Q = 1000.0*(350.0 - x[3]) + 100.0*err
        return np.clip(np.array([F, Q]), [10, -1e5], [500, 1e5]), err


class CSTRObserverDT:
    def __init__(self, mismatch_frac=0.0, seed=None):
        self.plant_model = CSTRPlant(seed=seed, mismatch_frac=mismatch_frac)
        self.K_o = np.array([[0.3,0,0],[0,0.3,0],[0,0,0.3],[0,0,0.1]])
        self.x_hat = np.array([0.5, 0.5, 350.0, 350.0])

    def reset(self, x0=None):
        self.x_hat = x0.copy() if x0 is not None else np.array([0.5,0.5,350.0,350.0])

    def step(self, u, y, sensor_mask=None):
        y_hat = self.x_hat[:3]
        innovation = y - y_hat
        if sensor_mask is not None:
            innovation = sensor_mask[:len(innovation)] * innovation
        dt = self.plant_model.dt
        k1 = self.plant_model._dynamics(self.x_hat, u)
        k2 = self.plant_model._dynamics(self.x_hat + 0.5*dt*k1, u)
        k3 = self.plant_model._dynamics(self.x_hat + 0.5*dt*k2, u)
        k4 = self.plant_model._dynamics(self.x_hat + dt*k3, u)
        self.x_hat = self.x_hat + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        self.x_hat += self.K_o @ innovation
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 3.0)
        self.x_hat[1] = np.clip(self.x_hat[1], 0.0, 3.0)
        self.x_hat[2] = np.clip(self.x_hat[2], 280.0, 450.0)
        self.x_hat[3] = np.clip(self.x_hat[3], 280.0, 420.0)
        return y_hat, self.x_hat.copy()

    def residual(self, y, y_hat, sensor_mask):
        return sensor_mask[:len(y)] * (y - y_hat)


# ── CSTR attacks (E_max=500, paper Section VII-B) ─────────────────

def _norm_energy(a, E_max):
    e = np.sum(a**2)
    if e > E_max: a *= np.sqrt(E_max / e)
    return a

def cstr_fdi_attack(H, rng, E_max=500.0):
    a = np.zeros((H, 3))
    t_start = 20 + rng.randint(0, 10)
    a[t_start:, 0] = 0.3 * rng.randn()
    return _norm_energy(a, E_max)

def cstr_thermal_attack(H, rng, E_max=500.0):
    a = np.zeros((H, 3))
    t_start = 20 + rng.randint(0, 10)
    t = np.arange(H); freq = 0.3 + 0.5*rng.rand()
    a[t_start:, 1] = 0.15 * np.sin(2*np.pi*freq*t[t_start:]*0.1)
    a[t_start:, 2] = 0.1 * np.cos(2*np.pi*freq*t[t_start:]*0.1 + rng.rand()*np.pi)
    return _norm_energy(a, E_max)

def cstr_concentration_attack(H, rng, E_max=500.0):
    a = np.zeros((H, 3))
    t_start = 20 + rng.randint(0, 10)
    ramp_len = H - t_start
    direction = rng.randn(2); direction /= np.linalg.norm(direction) + 1e-8
    for k in range(ramp_len):
        a[t_start+k, :2] = direction * (k/ramp_len) * 0.25
    return _norm_energy(a, E_max)

CSTR_ATTACK_GENERATORS = {
    'cstr_fdi': cstr_fdi_attack,
    'cstr_thermal': cstr_thermal_attack,
    'cstr_concentration': cstr_concentration_attack,
}


def run_cstr_episode(plant, twin, reference, attack_seq=None, sensor_mask=None):
    H = len(reference)
    if sensor_mask is None: sensor_mask = np.ones(plant.n_output)
    x0 = plant.reset(); twin.reset(x0.copy())
    residuals = np.zeros((H, plant.n_output))
    # FIX: quality z_k = [C_A, C_B, T] (3-vector, not scalar)
    z_traj = np.zeros((H, 3))
    integral_err = 0.0
    for k in range(H):
        CB_ref = reference[k, 0] if reference[k, 0] != 0 else 0.5
        u, err = CSTRPlant.pi_controller(plant.x, CB_ref, integral_err)
        integral_err += err * plant.dt
        a = attack_seq[k] if attack_seq is not None else None
        x_next, y = plant.step(u, a)
        # v2 FIX: sensor mask must gate the observer correction too (Eq. 4),
        # not only the residual. No effect for the full-sensor mask (ones).
        y_hat, x_hat = twin.step(u, y, sensor_mask)
        residuals[k] = twin.residual(y, y_hat, sensor_mask)
        z_traj[k] = x_next[:3]   # quality = [C_A, C_B, T]
    return {'residuals': residuals, 'z_traj': z_traj}


def generate_cstr_dataset(mismatch_frac=0.05, n_nom=20, n_atk=10, H=120, seed=42,
                          sensor_mask=None, return_stats=False):
    """Generate CSTR dataset: one window per episode, CUSUM legacy, W_z-weighted impact.

    v2: sensor_mask gates the detector path (observer correction + residual);
    return_stats additionally returns (nom_mean, nom_std) for the sensor study."""
    W_L = 8
    if sensor_mask is None:
        sensor_mask = np.ones(3)

    # ── Nominal episodes ──
    all_nom_residuals = []
    nom_windows = []
    for ep in range(n_nom):
        plant = CSTRPlant(seed=seed+100+ep)
        twin = CSTRObserverDT(mismatch_frac=mismatch_frac, seed=seed+1)
        ref = plant.reference(H)
        data = run_cstr_episode(plant, twin, ref, sensor_mask=sensor_mask)
        all_nom_residuals.append(data['residuals'])
        # One window per episode at mid-episode offset
        k0 = H//2 - W_L//2
        nom_windows.append(data['residuals'][k0:k0+W_L].flatten())

    nom_windows = np.nan_to_num(np.array(nom_windows), nan=0.0, posinf=0.0, neginf=0.0)
    nom_mean = nom_windows.mean(axis=0)
    nom_std = nom_windows.std(axis=0) + 1e-8
    nom_windows = (nom_windows - nom_mean) / nom_std

    # Estimate CUSUM baseline mu_0 from nominal residuals
    all_res = np.vstack(all_nom_residuals)
    norms = np.linalg.norm(all_res, axis=1)
    mu_0 = np.mean(norms) + 2 * np.std(norms)

    # ── Attack episodes ──
    attack_data = {}
    for aname, agen in CSTR_ATTACK_GENERATORS.items():
        atk_wins = []; impacts = []; legacy_responses = []; energies = []
        for ep in range(n_atk):
            plant = CSTRPlant(seed=seed+200+ep)
            twin = CSTRObserverDT(mismatch_frac=mismatch_frac, seed=seed+1)
            ref = plant.reference(H)
            a_seq = agen(H, np.random.RandomState(seed+300+ep), E_max=500.0)

            # Nominal for impact comparison
            plant_nom = CSTRPlant(seed=seed+200+ep)
            twin_nom = CSTRObserverDT(mismatch_frac=mismatch_frac, seed=seed+1)
            data_nom = run_cstr_episode(plant_nom, twin_nom, ref, sensor_mask=sensor_mask)

            data_atk = run_cstr_episode(plant, twin, ref, a_seq, sensor_mask)

            # FIX: impact with W_z = diag(1, 1, 0.01)
            impact = compute_impact(data_atk['z_traj'], data_nom['z_traj'], W_z=W_Z_CSTR)
            # FIX: CUSUM legacy detector
            leg_resp = cusum_legacy(data_atk['residuals'], mu_0=mu_0)
            energy = np.sum(a_seq**2)

            k0 = H//2 - W_L//2
            atk_wins.append(data_atk['residuals'][k0:k0+W_L].flatten())
            impacts.append(impact)
            legacy_responses.append(leg_resp)
            energies.append(energy)

        atk_arr = np.nan_to_num(np.array(atk_wins), nan=0.0, posinf=0.0, neginf=0.0)
        atk_arr = (atk_arr - nom_mean) / nom_std
        attack_data[aname] = {
            'windows': atk_arr, 'impacts': impacts,
            'legacy_responses': legacy_responses, 'energies': energies,
            'label': aname,
        }
    if return_stats:
        return nom_windows, attack_data, nom_mean, nom_std
    return nom_windows, attack_data
