"""
Surrogate-improvement ablation variants (reviewer R2.4; Supplementary Section M).

V0: current MLP (eigen-embedding targets, unconstrained range) -- baseline.
V1: normalized MLP -- unit-sphere embedding with squared inner product, so the
    surrogate kernel lies in [0,1] and k(x,x)=1 by construction (matches the
    fidelity-kernel range; models the fidelity of a rank-1 real embedding).
V2: larger capacity (D_e=64, trunk (128,64)).
V3: alignment-refined head -- V1 trunk frozen, linear head re-optimized to
    maximize centered kernel alignment (CKA) with the exact quantum Gram,
    prioritizing similarity structure over pointwise accuracy.
V4: expanded training pool -- V1 trained on the full window pool (all attack
    families + nominal) rather than the truncated subset used by V0.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _eigen_targets(K, embed_dim, regularize=1e-4):
    K = K + regularize * np.eye(len(K))
    w, U = np.linalg.eigh(K)
    idx = np.argsort(w)[::-1][:embed_dim]
    w = np.maximum(w[idx], 1e-8)
    return U[:, idx] * np.sqrt(w)[None, :]


class SurrogateVariant:
    """MLP surrogate with optional unit-norm/squared-inner-product kernel."""

    def __init__(self, embed_dim=15, hidden=(64, 32, 16), max_iter=500,
                 normalized=False, seed=0):
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.max_iter = max_iter
        self.normalized = normalized
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = None
        self.head_W = None  # optional alignment-refined head (V3)

    # -- training ---------------------------------------------------
    def fit(self, X_train, K_train):
        Y = _eigen_targets(K_train, self.embed_dim)
        if self.normalized:
            Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        Xs = self.scaler.fit_transform(X_train)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden, max_iter=self.max_iter,
            early_stopping=True, validation_fraction=0.15, random_state=self.seed,
            learning_rate='adaptive', alpha=1e-3,
            batch_size=min(32, max(2, len(X_train) // 2)))
        self.model.fit(Xs, Y)
        return self

    # -- embeddings / kernel ----------------------------------------
    def _hidden_activations(self, X):
        """Manual forward pass up to the last hidden layer."""
        Xs = self.scaler.transform(X)
        A = Xs
        n_hidden = len(self.model.coefs_) - 1
        for l in range(n_hidden):
            A = np.maximum(0.0, A @ self.model.coefs_[l] + self.model.intercepts_[l])
        return A

    def embed(self, X):
        if self.head_W is not None:
            E = self._hidden_activations(X) @ self.head_W.T
        else:
            E = self.model.predict(self.scaler.transform(X))
            if E.ndim == 1:
                E = E[:, None]
        if self.normalized:
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        return E

    def kernel(self, X, Y=None):
        EX = self.embed(X)
        EY = EX if Y is None else self.embed(Y)
        G = EX @ EY.T
        if self.normalized:
            G = G ** 2          # fidelity of the rank-1 embedding, in [0,1]
        return G

    # -- V3: alignment-refined linear head ---------------------------
    def refine_head_cka(self, X_train, K_train, maxiter=60, seed=0):
        """Re-fit the last linear layer to maximize CKA(K_surr, K_exact)."""
        H = self._hidden_activations(X_train)          # (N, h)
        h = H.shape[1]
        n = len(K_train)
        C = np.eye(n) - np.ones((n, n)) / n
        Kc = C @ K_train @ C
        Kc_norm = np.sqrt(np.sum(Kc * Kc)) + 1e-12
        rng = np.random.RandomState(seed)
        # warm start from the trained output layer (transposed)
        W0 = self.model.coefs_[-1].T + 0.01 * rng.randn(self.embed_dim, h)

        def neg_cka(w):
            W = w.reshape(self.embed_dim, h)
            E = H @ W.T
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            G = (E @ E.T) ** 2
            Gc = C @ G @ C
            num = np.sum(Gc * Kc)
            den = Kc_norm * (np.sqrt(np.sum(Gc * Gc)) + 1e-12)
            return -num / den

        res = minimize(neg_cka, W0.ravel(), method='L-BFGS-B',
                       options=dict(maxiter=maxiter))
        self.head_W = res.x.reshape(self.embed_dim, h)
        self.normalized = True
        return self

    # -- metrics -----------------------------------------------------
    def max_error(self, X_test, K_exact_test):
        G = self.kernel(X_test)
        return float(np.max(np.abs(G - K_exact_test)))

    def mae(self, X_test, K_exact_test):
        G = self.kernel(X_test)
        iu = np.triu_indices(len(G), k=1)
        return float(np.mean(np.abs(G[iu] - K_exact_test[iu])))
