"""
Learned surrogate distillation: MLP network to approximate the quantum kernel.

Trains φ_ω: R^d → R^D such that φ_ω(x)^T φ_ω(y) ≈ k_θ(x,y).
Uses scikit-learn MLPRegressor on Cholesky factors of the kernel matrix.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class LearnedSurrogate:
    """MLP-based surrogate embedding for the quantum kernel."""

    def __init__(self, embed_dim=20, hidden=(64, 32, 16), max_iter=800):
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_train, kernel_fn, regularize=1e-4):
        """
        Train surrogate: K = UΛU^T → targets Y = U·sqrt(Λ) → MLP: X → Y.
        """
        K = kernel_fn(X_train, X_train)
        K += regularize * np.eye(len(K))
        
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1][:self.embed_dim]
        eigvals = np.maximum(eigvals[idx], 1e-8)
        eigvecs = eigvecs[:, idx]
        Y_target = eigvecs * np.sqrt(eigvals)[None, :]
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
            learning_rate='adaptive',
            alpha=1e-3,
            batch_size=min(32, max(2, len(X_train) // 2)),
        )
        self.model.fit(X_scaled, Y_target)
        self.fitted = True
        return self

    def embed(self, X):
        """Compute surrogate embedding φ_ω(X)."""
        assert self.fitted
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def surrogate_kernel(self, phi_X, phi_Y=None):
        if phi_Y is None:
            return phi_X @ phi_X.T
        return phi_X @ phi_Y.T

    def approximation_error(self, X, kernel_fn, n_pairs=200):
        """Estimate ε_surr = max |k_θ(x,x') - φ(x)^T φ(x')|."""
        rng = np.random.RandomState(0)
        N = len(X)
        phi = self.embed(X)
        max_err = 0.0
        for _ in range(n_pairs):
            i, j = rng.randint(0, N, size=2)
            k_exact = kernel_fn(X[i:i+1], X[j:j+1])[0, 0]
            k_surr = phi[i] @ phi[j]
            max_err = max(max_err, abs(k_exact - k_surr))
        return max_err
