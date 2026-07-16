"""
Strengthened non-kernel baselines for revision v2 (reviewer R2.7 / R3.1).

Tuned One-Class SVM, tuned Isolation Forest (validation-grid selection per
benchmark and seed, mirroring the classical-kernel bandwidth protocol), and a
window-input autoencoder residual detector. Protocol: Supplementary Section N.
"""
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

OCSVM_GRID = [dict(nu=nu, gamma=g)
              for nu in (0.01, 0.05, 0.1, 0.2, 0.5)
              for g in ('scale', 0.01, 0.1, 1.0)]

IFOREST_GRID = [dict(n_estimators=ne, max_samples=ms, contamination=c)
                for ne in (100, 200)
                for ms in (0.5, 1.0)
                for c in ('auto', 0.05, 0.1)]


def _select(model_fn, grid, Xn, Xa_tune):
    """Pick the grid point with the best nominal-vs-tune-attack AUC.

    Selection uses the first half of the pooled attack windows (Xa_tune),
    per benchmark and seed -- NOT per attack family -- matching the
    deployment assumption that the family is unknown (Supp. N).
    """
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xa_tune))])
    X = np.vstack([Xn, Xa_tune])
    best_auc, best_params, best_model = -np.inf, grid[0], None
    for params in grid:
        model = model_fn(**params)
        model.fit(Xn)
        s = -model.decision_function(X)
        try:
            auc = roc_auc_score(y, s)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_params, best_model = auc, params, model
    return best_model, best_params, best_auc


def fit_tuned_ocsvm(Xn, Xa_tune):
    return _select(lambda **p: OneClassSVM(kernel='rbf', **p),
                   OCSVM_GRID, Xn, Xa_tune)


def fit_tuned_iforest(Xn, Xa_tune, seed=42):
    return _select(lambda **p: IsolationForest(random_state=seed, **p),
                   IFOREST_GRID, Xn, Xa_tune)


class WindowAutoencoder:
    """Symmetric MLP autoencoder on PCA features (Supp. N).

    Encoder (d -> 32 -> 8), decoder mirrored; trained on nominal windows only
    (attack-agnostic); anomaly score = squared reconstruction error.
    With m=20 nominal windows, early stopping is disabled (validation split
    of 3 windows is too unstable); fixed max_iter with L2 regularization.
    """

    def __init__(self, seed=0, max_iter=2000):
        self.model = MLPRegressor(hidden_layer_sizes=(32, 8, 32),
                                  activation='relu', solver='adam',
                                  learning_rate_init=1e-3, alpha=1e-3,
                                  max_iter=max_iter, random_state=seed,
                                  early_stopping=False)

    def fit(self, Xn):
        self.model.fit(Xn, Xn)
        return self

    def score(self, X):
        R = self.model.predict(X)
        return np.sum((X - R) ** 2, axis=1)


def eval_scored_baseline(score_fn, Xn, Xa, alpha=0.05):
    """Nominal-quantile-thresholded per-window F1 (mirrors kernel protocol)."""
    from sklearn.metrics import f1_score
    s_nom = score_fn(Xn)
    thresh = np.percentile(s_nom, 100 * (1 - alpha))
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xa))])
    s = np.concatenate([s_nom, score_fn(Xa)])
    return f1_score(y, (s > thresh).astype(int), zero_division=0)
