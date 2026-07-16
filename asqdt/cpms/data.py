"""
Data generation pipeline: run episodes, extract block-independent windows.
Enforces Assumption 6: one window per episode at fixed mid-episode offset.
"""
import numpy as np
from ..cpms.plant import (CPMSPlant, ObserverDT, ATTACK_GENERATORS,
                           legacy_detector_response, compute_impact)
from .. import config as cfg


def run_episode(plant, twin, reference, attack_seq=None, sensor_mask=None):
    """Run one transient episode. Returns trajectory data."""
    H = len(reference)
    if sensor_mask is None:
        sensor_mask = np.ones(cfg.N_OUTPUT)

    x0 = plant.reset()
    twin.reset(x0.copy())

    residuals = np.zeros((H, cfg.N_OUTPUT))
    x_traj = np.zeros((H, cfg.N_STATE))
    z_traj = np.zeros((H, cfg.N_STATE))   # full state as quality (CNC: W_z=I_6)
    z_nom  = np.zeros((H, cfg.N_STATE))
    u_traj = np.zeros((H, cfg.N_INPUT))

    for k in range(H):
        p_ref = reference[k]
        u = CPMSPlant.pd_controller(plant.x, p_ref)
        a = attack_seq[k] if attack_seq is not None else None
        x_next, y = plant.step(u, a)
        y_hat, x_hat = twin.step(u, y, sensor_mask)

        residuals[k] = twin.residual(y, y_hat, sensor_mask)
        x_traj[k] = x_next
        z_traj[k] = x_next      # quality z_k = x_k (paper: CNC uses full state)
        u_traj[k] = u

    return {
        'residuals': residuals,
        'x_traj': x_traj,
        'z_traj': z_traj,
        'u_traj': u_traj,
    }


def extract_one_window(residuals, W_L):
    """
    Extract ONE window per episode, centered at mid-episode.
    k0 = floor(H/2) - floor(W_L/2)   (paper Section VII-B)
    This satisfies Assumption 6 (block independence).
    """
    H = len(residuals)
    k0 = H // 2 - W_L // 2
    k0 = max(0, min(k0, H - W_L))  # safety clamp
    return residuals[k0:k0 + W_L].flatten()


def extract_sliding_windows(residuals, W_L, stride=1):
    """
    Extract overlapping sliding windows for detection metrics.
    These do NOT satisfy block-independence — empirical use only.
    """
    H = len(residuals)
    windows = []
    for start in range(0, H - W_L + 1, stride):
        windows.append(residuals[start:start + W_L].flatten())
    return np.array(windows) if windows else np.zeros((0, W_L * residuals.shape[1]))


def generate_dataset(mismatch_frac=0.05, sensor_mask=None, seed=cfg.SEED,
                     W_L=cfg.WINDOW_LEN, E_max=50.0, legacy_method='chi_sq'):
    """
    Generate complete CNC dataset with one window per episode.

    Returns:
        nominal_windows: (m, d) — one window per nominal episode
        attack_data: dict[name] -> {windows, impacts, legacy_responses, ...}
        nom_Sigma_r_inv: estimated nominal residual inverse covariance
    """
    rng = np.random.RandomState(seed)

    if sensor_mask is None:
        sensor_mask = np.ones(cfg.N_OUTPUT)

    # ── Collect nominal residuals for covariance estimation ────────
    all_nom_residuals = []
    nom_windows = []
    for ep in range(cfg.N_EPISODES_NOM):
        plant_ep = CPMSPlant(seed=seed + 100 + ep)
        twin_ep = ObserverDT(mismatch_frac=mismatch_frac, seed=seed + 1)
        ref = plant_ep.reference(cfg.HORIZON)
        data = run_episode(plant_ep, twin_ep, ref, sensor_mask=sensor_mask)
        all_nom_residuals.append(data['residuals'])
        nom_windows.append(extract_one_window(data['residuals'], W_L))

    nominal_windows = np.array(nom_windows)  # (m, W_L*p)

    # Estimate nominal residual covariance for chi-squared legacy detector
    all_res = np.vstack(all_nom_residuals)
    Sigma_r = np.cov(all_res.T) + 1e-6 * np.eye(all_res.shape[1])
    try:
        Sigma_r_inv = np.linalg.inv(Sigma_r)
    except np.linalg.LinAlgError:
        Sigma_r_inv = np.eye(all_res.shape[1])

    # Standardize features
    nom_mean = nominal_windows.mean(axis=0)
    nom_std = nominal_windows.std(axis=0) + 1e-8
    nominal_windows = (nominal_windows - nom_mean) / nom_std

    # ── Attack episodes ───────────────────────────────────────────
    attack_data = {}
    for atk_name, atk_gen in ATTACK_GENERATORS.items():
        atk_windows = []
        impacts = []
        legacy_responses = []
        energies = []

        for ep in range(cfg.N_EPISODES_ATK):
            plant_ep = CPMSPlant(seed=seed + 200 + ep)
            twin_ep = ObserverDT(mismatch_frac=mismatch_frac, seed=seed + 1)
            ref = plant_ep.reference(cfg.HORIZON)

            atk_rng = np.random.RandomState(seed + 300 + ep)
            a_seq = atk_gen(cfg.HORIZON, atk_rng, E_max=E_max)

            # Run nominal (same seed) for impact comparison
            plant_nom = CPMSPlant(seed=seed + 200 + ep)
            twin_nom = ObserverDT(mismatch_frac=mismatch_frac, seed=seed + 1)
            data_nom = run_episode(plant_nom, twin_nom, ref, sensor_mask=sensor_mask)

            # Run attacked
            data_atk = run_episode(plant_ep, twin_ep, ref, a_seq, sensor_mask)

            impact = compute_impact(data_atk['z_traj'], data_nom['z_traj'])
            leg_resp = legacy_detector_response(data_atk['residuals'],
                                                 method=legacy_method,
                                                 Sigma_r_inv=Sigma_r_inv)
            energy = np.sum(a_seq ** 2)

            atk_windows.append(extract_one_window(data_atk['residuals'], W_L))
            impacts.append(impact)
            legacy_responses.append(leg_resp)
            energies.append(energy)

        atk_win_arr = np.array(atk_windows)
        atk_win_arr = (np.nan_to_num(atk_win_arr) - nom_mean) / nom_std

        attack_data[atk_name] = {
            'windows': atk_win_arr,
            'impacts': impacts,
            'legacy_responses': legacy_responses,
            'energies': energies,
            'label': atk_name,
        }

    return nominal_windows, attack_data, Sigma_r_inv, nom_mean, nom_std
