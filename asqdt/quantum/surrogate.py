"""
Nyström surrogate: distill quantum kernel into finite-dim classical embedding.

Given k_θ and a representative sample Z = {z_1,...,z_R}, the Nyström embedding is:
  φ̃(ξ) = Λ^{-1/2} U^T k_θ(Z, ξ)
where K_ZZ = U Λ U^T is the eigendecomposition of the kernel on landmarks.

Guarantees: |k_θ(ξ,ξ') - φ̃(ξ)^T φ̃(ξ')| ≤ ε_surr  on compact B.
"""
import numpy as np
from .. import config as cfg


class NystromSurrogate:
    """Classical surrogate embedding via Nyström approximation."""

    def __init__(self, rank=cfg.NYSTROM_RANK):
        self.rank = rank
        self.landmarks = None
        self.U = None
        self.Lambda_inv_sqrt = None
        self.fitted = False

    def fit(self, X_landmarks, kernel_fn):
        """
        Fit surrogate from landmark points and kernel function.
        
        Args:
            X_landmarks: (R, d) array of representative points
            kernel_fn: callable(X, Y) -> (len(X), len(Y)) kernel matrix
        """
        self.landmarks = X_landmarks
        R = len(X_landmarks)
        K_ZZ = kernel_fn(X_landmarks, X_landmarks)
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_ZZ)
        # Keep top `rank` components (sorted ascending, so take last)
        idx = np.argsort(eigenvalues)[::-1][:self.rank]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Regularize small eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        self.Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        self.U = eigenvectors
        self.fitted = True

    def embed(self, X, kernel_fn):
        """
        Compute surrogate embedding φ̃(X).
        
        Args:
            X: (N, d) array of points
            kernel_fn: callable(X, Y) -> kernel matrix
        Returns:
            (N, rank) embedding
        """
        assert self.fitted, "Call fit() first"
        K_xz = kernel_fn(X, self.landmarks)  # (N, R)
        # φ̃(x) = Λ^{-1/2} U^T k(Z, x)
        return K_xz @ self.U @ self.Lambda_inv_sqrt  # (N, rank)

    def surrogate_kernel(self, phi_X, phi_Y=None):
        """Compute surrogate kernel from embeddings: k̃(x,y) = φ̃(x)^T φ̃(y)."""
        if phi_Y is None:
            return phi_X @ phi_X.T
        return phi_X @ phi_Y.T

    def approximation_error(self, X, kernel_fn, n_pairs=200):
        """Estimate ε_surr = max |k_θ(x,x') - k̃(x,x')| on sample."""
        rng = np.random.RandomState(0)
        N = len(X)
        phi_X = self.embed(X, kernel_fn)
        max_err = 0.0
        for _ in range(n_pairs):
            i, j = rng.randint(0, N, size=2)
            k_exact = kernel_fn(X[i:i+1], X[j:j+1])[0, 0]
            k_surr = phi_X[i] @ phi_X[j]
            max_err = max(max_err, abs(k_exact - k_surr))
        return max_err
