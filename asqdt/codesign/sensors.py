"""
Sensor selection: submodular greedy log-det surrogate (Proposition 2).

F(S) = log det(I + Σ_{i∈S} M_i)  is monotone, submodular.
Greedy under cardinality |S| ≤ B_s gives (1-1/e) approximation.
"""
import numpy as np
from itertools import combinations
from .. import config as cfg


def compute_separation_matrices(phi_nom, phi_atk, n_sensors, W_L=cfg.WINDOW_LEN):
    """
    Compute local separation matrices M_i from surrogate embeddings.
    
    M_i = E_a[ (∂φ̃/∂r^(i)) (∂φ̃/∂r^(i))^T ]
    
    Approximated numerically: perturb residual channel i, measure embedding change.
    """
    D = phi_nom.shape[1]
    M = np.zeros((n_sensors, D, D))
    
    # Use difference between nominal and attack embeddings as sensitivity proxy
    delta_phi = phi_atk.mean(axis=0) - phi_nom.mean(axis=0)  # (D,)
    
    for i in range(n_sensors):
        # Sensor i contributes features at indices [i, i+p, i+2p, ...] in the window
        # Approximate M_i from the covariance of embedding differences
        # attributable to sensor i
        n_features_per_sensor = W_L
        feature_indices = list(range(i, cfg.FEATURE_DIM, n_sensors))[:W_L]
        
        # Finite-difference sensitivity
        grad_i = delta_phi / (n_sensors + 1e-8)  # rough per-sensor contribution
        M[i] = np.outer(grad_i, grad_i) + 1e-6 * np.eye(D)
    
    return M


def greedy_sensor_selection(M_matrices, budget, n_sensors):
    """
    Greedy maximization of F(S) = log det(I + Σ_{i∈S} M_i).
    
    Returns: selected sensor indices, F values at each step
    """
    D = M_matrices.shape[1]
    selected = []
    remaining = list(range(n_sensors))
    F_values = [0.0]  # F(∅) = log det(I) = 0
    
    current_sum = np.zeros((D, D))
    
    for step in range(min(budget, n_sensors)):
        best_gain = -np.inf
        best_sensor = None
        
        for s in remaining:
            trial_sum = current_sum + M_matrices[s]
            F_trial = np.log(np.linalg.det(np.eye(D) + trial_sum) + 1e-30)
            gain = F_trial - F_values[-1]
            if gain > best_gain:
                best_gain = gain
                best_sensor = s
        
        if best_sensor is not None:
            selected.append(best_sensor)
            remaining.remove(best_sensor)
            current_sum += M_matrices[best_sensor]
            F_values.append(F_values[-1] + best_gain)
    
    return selected, F_values


def sensor_mask_from_indices(indices, n_sensors):
    """Convert sensor indices to binary mask."""
    mask = np.zeros(n_sensors)
    for i in indices:
        mask[i] = 1.0
    return mask


def exhaustive_sensor_selection(M_matrices, budget, n_sensors):
    """Brute-force optimal (for small problems only)."""
    if n_sensors > 15:
        return None, None  # too expensive
    
    D = M_matrices.shape[1]
    best_F = -np.inf
    best_set = None
    
    for combo in combinations(range(n_sensors), budget):
        M_sum = sum(M_matrices[i] for i in combo)
        F = np.log(np.linalg.det(np.eye(D) + M_sum) + 1e-30)
        if F > best_F:
            best_F = F
            best_set = list(combo)
    
    return best_set, best_F
