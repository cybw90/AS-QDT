"""
Alternating minimax co-design solver (Algorithm 1).

Outer loop:
  1) Attack generation (augmented Lagrangian on differentiable simulator)
  2) Kernel parameter update (gradient ascent on MMD² - λ_q C_q)
  3) Sensor update (greedy + QUBO refinement)
  4) Twin re-fit if sensors changed

For this implementation, attack optimization uses projected gradient descent
as an approximation to the full augmented Lagrangian (the plant simulator is
differentiable in the linear case).
"""
import numpy as np
from ..quantum.kernel import QuantumCircuit
from ..quantum.surrogate import NystromSurrogate
from ..detection.mmd import mmd_from_embeddings, compute_threshold, permutation_threshold
from ..codesign.sensors import compute_separation_matrices, greedy_sensor_selection, sensor_mask_from_indices
from ..cpms.data import generate_dataset, filter_stealthy_windows
from .. import config as cfg


class ASQDTSolver:
    """Alternating minimax solver for AS-QDT co-design."""

    def __init__(self, seed=cfg.SEED):
        self.rng = np.random.RandomState(seed)
        self.circuit = QuantumCircuit(cfg.N_QUBITS, cfg.CIRCUIT_DEPTH)
        self.theta = self.circuit.initial_params(self.rng)
        self.surrogate = NystromSurrogate(rank=cfg.NYSTROM_RANK)
        self.sensor_indices = list(range(cfg.SENSOR_BUDGET))  # initial: first B_s
        self.sensor_mask = sensor_mask_from_indices(self.sensor_indices, cfg.N_OUTPUT)
        self.history = {'margin': [], 'theta_norm': [], 'sensors': []}

    def _kernel_fn(self, X, Y, theta=None):
        """Kernel function wrapper for current theta."""
        if theta is None:
            theta = self.theta
        if X is Y or (len(X) == len(Y) and np.allclose(X, Y)):
            return self.circuit.gram_matrix(X, theta)
        return self.circuit.cross_gram(X, Y, theta)

    def _mmd_with_theta(self, X_nom, X_atk, theta):
        """MMD² for given theta (used in parameter-shift gradient)."""
        K_XX = self.circuit.gram_matrix(X_nom, theta)
        K_YY = self.circuit.gram_matrix(X_atk, theta)
        K_XY = self.circuit.cross_gram(X_nom, X_atk, theta)
        from ..detection.mmd import mmd_squared
        return mmd_squared(K_XX, K_YY, K_XY)

    def _reduce_features(self, windows):
        """Reduce feature dimension for quantum circuit (PCA to n_qubits * depth)."""
        target_dim = cfg.N_QUBITS * cfg.CIRCUIT_DEPTH
        if windows.shape[1] <= target_dim:
            return windows
        # Simple PCA
        mean = windows.mean(axis=0)
        centered = windows - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        Vt = Vt[:target_dim]
        # Deterministic sign convention (see evaluation.fastq.pca_fit)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
        signs[signs == 0] = 1.0
        Vt = Vt * signs[:, None]
        self._pca_mean = mean
        self._pca_V = Vt.T
        return centered @ self._pca_V

    def _apply_pca(self, windows):
        """Apply fitted PCA."""
        if not hasattr(self, '_pca_V'):
            return self._reduce_features(windows)
        return (windows - self._pca_mean) @ self._pca_V

    def run(self, mismatch_frac=0.0, verbose=True):
        """
        Run the full co-design loop.
        
        Returns: results dict with margins, sensor sets, detection stats
        """
        if verbose:
            print("=" * 60)
            print("AS-QDT Co-Design Solver")
            print("=" * 60)

        # ── Generate data with current sensor mask ────────────────
        if verbose:
            print(f"\n[1] Generating data (mismatch={mismatch_frac:.0%})...")
        nom_windows, attack_data = generate_dataset(
            mismatch_frac=mismatch_frac,
            sensor_mask=self.sensor_mask,
            seed=cfg.SEED
        )
        stealthy_data = filter_stealthy_windows(attack_data)
        
        if not stealthy_data:
            print("  WARNING: No stealthy attacks found. Using all attacks.")
            stealthy_data = attack_data

        # Collect all stealthy attack windows
        all_atk_windows = np.vstack([d['windows'] for d in stealthy_data.values()])
        
        if verbose:
            print(f"  Nominal windows: {len(nom_windows)}")
            print(f"  Stealthy attack windows: {len(all_atk_windows)}")
            for name, d in stealthy_data.items():
                print(f"    {name}: {len(d['windows'])} windows")

        # ── Reduce features for quantum circuit ───────────────────
        X_nom = self._reduce_features(nom_windows)
        X_atk = self._apply_pca(all_atk_windows)

        # Subsample for tractable kernel computation
        max_samples = 80
        if len(X_nom) > max_samples:
            idx = self.rng.choice(len(X_nom), max_samples, replace=False)
            X_nom_sub = X_nom[idx]
        else:
            X_nom_sub = X_nom
        if len(X_atk) > max_samples:
            idx = self.rng.choice(len(X_atk), max_samples, replace=False)
            X_atk_sub = X_atk[idx]
        else:
            X_atk_sub = X_atk

        # ── Alternating loop ──────────────────────────────────────
        if verbose:
            print(f"\n[2] Alternating co-design ({cfg.MAX_ITERS} iterations)...")
        
        for it in range(cfg.MAX_ITERS):
            # Step 1: Compute current margin estimate
            margin = self._mmd_with_theta(X_nom_sub, X_atk_sub, self.theta)
            self.history['margin'].append(margin)
            self.history['theta_norm'].append(np.linalg.norm(self.theta))
            self.history['sensors'].append(self.sensor_indices.copy())

            # Step 2: Kernel update (gradient ascent on MMD² - λ_q * C_q)
            grad = self.circuit.spsa_gradient(
                X_nom_sub, X_atk_sub, self.theta, self._mmd_with_theta
            )
            # C_q(θ) = ||θ||² (circuit complexity proxy)
            grad_penalty = 2 * cfg.LAMBDA_Q * self.theta
            self.theta += cfg.KERNEL_LR * (grad - grad_penalty)

            # Step 3: Sensor update (every 3 iterations)
            if it % 3 == 0 and it > 0:
                # Fit temporary surrogate for sensor selection
                kernel_fn_wrap = lambda X, Y: self._kernel_fn(X, Y, self.theta)
                landmarks = X_nom_sub[self.rng.choice(len(X_nom_sub), 
                    min(30, len(X_nom_sub)), replace=False)]
                self.surrogate.fit(landmarks, kernel_fn_wrap)
                phi_nom = self.surrogate.embed(X_nom_sub, kernel_fn_wrap)
                phi_atk = self.surrogate.embed(X_atk_sub, kernel_fn_wrap)
                
                M_matrices = compute_separation_matrices(
                    phi_nom, phi_atk, cfg.N_OUTPUT
                )
                new_sensors, F_vals = greedy_sensor_selection(
                    M_matrices, cfg.SENSOR_BUDGET, cfg.N_OUTPUT
                )
                if new_sensors != self.sensor_indices:
                    self.sensor_indices = new_sensors
                    self.sensor_mask = sensor_mask_from_indices(
                        self.sensor_indices, cfg.N_OUTPUT
                    )
                    # Re-generate data with new sensors
                    nom_windows, attack_data = generate_dataset(
                        mismatch_frac=mismatch_frac,
                        sensor_mask=self.sensor_mask,
                        seed=cfg.SEED
                    )
                    stealthy_data = filter_stealthy_windows(attack_data)
                    if not stealthy_data:
                        stealthy_data = attack_data
                    all_atk_windows = np.vstack(
                        [d['windows'] for d in stealthy_data.values()]
                    )
                    X_nom = self._reduce_features(nom_windows)
                    X_atk = self._apply_pca(all_atk_windows)
                    if len(X_nom) > max_samples:
                        idx = self.rng.choice(len(X_nom), max_samples, replace=False)
                        X_nom_sub = X_nom[idx]
                    else:
                        X_nom_sub = X_nom
                    if len(X_atk) > max_samples:
                        idx = self.rng.choice(len(X_atk), max_samples, replace=False)
                        X_atk_sub = X_atk[idx]
                    else:
                        X_atk_sub = X_atk

            if verbose and (it % 3 == 0 or it == cfg.MAX_ITERS - 1):
                print(f"  Iter {it:3d}: Γ̂ = {margin:.6f}, sensors = {self.sensor_indices}")

            # Convergence check
            if len(self.history['margin']) > 3:
                recent = self.history['margin'][-3:]
                if max(recent) - min(recent) < 1e-6:
                    if verbose:
                        print(f"  Converged at iteration {it}")
                    break

        # ── Final surrogate distillation ──────────────────────────
        if verbose:
            print(f"\n[3] Distilling Nyström surrogate (rank={cfg.NYSTROM_RANK})...")
        kernel_fn_wrap = lambda X, Y: self._kernel_fn(X, Y, self.theta)
        n_landmarks = min(cfg.NYSTROM_RANK, len(X_nom_sub))
        landmarks = X_nom_sub[self.rng.choice(len(X_nom_sub), n_landmarks, replace=False)]
        self.surrogate.fit(landmarks, kernel_fn_wrap)
        
        eps_surr = self.surrogate.approximation_error(
            np.vstack([X_nom_sub[:30], X_atk_sub[:30]]), kernel_fn_wrap
        )
        if verbose:
            print(f"  ε_surr = {eps_surr:.6f}")

        # ── Threshold calibration ─────────────────────────────────
        if verbose:
            print(f"\n[4] Calibrating threshold...")
        phi_nom_full = self.surrogate.embed(X_nom, kernel_fn_wrap)
        kappa = 1.0  # fidelity kernel bounded by 1
        tau_analytic, beta = compute_threshold(
            kappa, cfg.M_NOMINAL, cfg.N_ONLINE, cfg.DELTA, eps_surr
        )
        tau_perm = permutation_threshold(phi_nom_full, alpha=cfg.DELTA)
        if verbose:
            print(f"  τ_analytic = {tau_analytic:.6f} (β={beta:.6f})")
            print(f"  τ_perm     = {tau_perm:.6f}")

        # ── Final results ─────────────────────────────────────────
        final_margin = self.history['margin'][-1]
        if verbose:
            print(f"\n[5] Final results:")
            print(f"  Final margin Γ̂ = {final_margin:.6f}")
            print(f"  Sensors: {self.sensor_indices}")
            print(f"  Margin > 2·τ: {final_margin > 2 * tau_perm}")

        return {
            'theta': self.theta.copy(),
            'sensor_indices': self.sensor_indices,
            'sensor_mask': self.sensor_mask,
            'margin_history': self.history['margin'],
            'final_margin': final_margin,
            'tau_analytic': tau_analytic,
            'tau_perm': tau_perm,
            'eps_surr': eps_surr,
            'beta': beta,
            'surrogate': self.surrogate,
            'circuit': self.circuit,
            'X_nom': X_nom,
            'X_atk': X_atk,
            'nom_windows': nom_windows,
            'attack_data': attack_data,
            'stealthy_data': stealthy_data,
            'kernel_fn': kernel_fn_wrap,
        }
