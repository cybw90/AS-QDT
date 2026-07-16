"""
Sensor co-design v2 (reviewers R2.5 / R4.5; Supplementary Section P).

FIXES a bug in compute_separation_matrices (sensors.py): the v1 implementation
assigned the SAME rank-1 matrix M_i to every sensor (the per-sensor Jacobian
was never actually computed), making greedy selection degenerate. This module
implements Definition 6 faithfully:

    M_i = E_a[ J_i(xi_bar) J_i(xi_bar)^T ],
    J_i = d(phi_tilde o Pi o standardize)/d xi_bar^(i)  in R^{D_e x W_L},

estimated by finite differences of the composite map on attack-perturbed
windows pooled across all families. Feature layout: flattened (W_L, p)
windows, so sensor i occupies raw indices {k*p + i : k = 0..W_L-1}.
"""
import numpy as np


def separation_matrices_v2(raw_windows_std, embed_fn, pca_mu, pca_V,
                           n_sensors, W_L, h=1e-2):
    """M_i from finite-difference Jacobians of embed(PCA(window)).

    Args:
        raw_windows_std: (N, W_L*p) attack windows in standardized raw-feature
            space (i.e., before PCA -- the space where sensor blocks live).
        embed_fn: maps PCA features (B, d) -> embeddings (B, D_e).
        pca_mu, pca_V: the frozen PCA (features = (x - mu) @ V).
        n_sensors: p; W_L: window length; h: finite-difference step.
    Returns: (p, D_e, D_e) array of M_i.
    """
    N, F = raw_windows_std.shape
    assert F == W_L * n_sensors
    base = embed_fn((raw_windows_std - pca_mu) @ pca_V)          # (N, D_e)
    D_e = base.shape[1]
    M = np.zeros((n_sensors, D_e, D_e))
    for i in range(n_sensors):
        cols = [k * n_sensors + i for k in range(W_L)]
        # Batch all (window, column) perturbations for sensor i
        P = np.repeat(raw_windows_std, W_L, axis=0)              # (N*W_L, F)
        for r, c in enumerate(cols):
            P[r::W_L, c] += h
        E = embed_fn((P - pca_mu) @ pca_V)                       # (N*W_L, D_e)
        J = (E - np.repeat(base, W_L, axis=0)) / h               # columns of J_i
        J = J.reshape(N, W_L, D_e)                               # (N, W_L, D_e)
        # M_i = E_n[ J_i J_i^T ] with J_i in R^{D_e x W_L}
        M[i] = np.einsum('nwd,nwe->de', J, J) / N
        M[i] += 1e-9 * np.eye(D_e)
    return M


def greedy_logdet(M, budget):
    """Greedy maximization of F(S) = logdet(I + sum_{i in S} M_i)."""
    p, D, _ = M.shape
    selected, remaining = [], list(range(p))
    cur = np.zeros((D, D))
    F_prev = 0.0
    for _ in range(min(budget, p)):
        best_gain, best_s = -np.inf, None
        for s in remaining:
            sign, logdet = np.linalg.slogdet(np.eye(D) + cur + M[s])
            gain = logdet - F_prev
            if gain > best_gain:
                best_gain, best_s = gain, s
        selected.append(best_s)
        remaining.remove(best_s)
        cur += M[best_s]
        F_prev += best_gain
    return sorted(selected)


def top_variance_sensors(nom_std_raw, n_sensors, W_L, budget):
    """Attack-agnostic baseline: B_s sensors with largest nominal residual
    variance, computed from the per-feature stds of the raw nominal windows."""
    var = np.zeros(n_sensors)
    for i in range(n_sensors):
        cols = [k * n_sensors + i for k in range(W_L)]
        var[i] = np.mean(nom_std_raw[cols] ** 2)
    return sorted(np.argsort(var)[::-1][:budget].tolist())


def mask_from(indices, p):
    m = np.zeros(p)
    m[list(indices)] = 1.0
    return m
