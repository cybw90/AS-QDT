"""
External validation on public ICS datasets (SWaT / WADI / HAI / BATADAL).

Detector-component-only validation, mirroring the paper's corrected held-out
protocol (Supplementary L): because public traces ship no executable plant
model, twin residualization is replaced by a learned one-step forecaster and
attack-set membership cannot be verified. This tests ONLY (i) whether the
corrected calibration protocol behaves on real ICS residuals and (ii) whether
fidelity-kernel concentration reproduces outside the simulators. It does not
exercise the co-design loop and makes no SOTA claim (framed accordingly in
the paper's SWaT/WADI paragraph).

Protocol
  1. Numeric sensor channels; median-downsample by factor `ds`.
  2. Multi-output ridge forecaster on `n_lags` lags, fit on the first
     `train_frac` of the NORMAL trace; residuals r_t = y_t - yhat_t.
  3. Non-overlapping residual windows of length W_L. From post-train normal,
     chronologically: m=20 reference windows, 30 calibration, 30 test
     negatives (disjoint). One window per labeled ATTACK scenario (segment
     center) = positives; scenarios shorter than a window are skipped.
  4. Standardize by reference stats; PCA d=12 fit on reference.
  5. Detectors fitted on reference only; threshold = 95% quantile of
     calibration scores; report pooled AUC / F1 / FPR + per-scenario hits.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf, laplacian_kernel as sk_lap
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import OneClassSVM

from .. import config as cfg
from ..quantum.kernel import QuantumCircuit
from ..evaluation.fastq import (QKernelCache, gram_from_sv, pca_fit,
                                design_quantum_kernel, rbf_bandwidth)
from ..evaluation.baselines_v2 import (fit_tuned_ocsvm, fit_tuned_iforest,
                                       WindowAutoencoder)

LABEL_ALIASES = ('normal/attack', 'normal_attack', 'label', 'attack')
ATTACK_TOKENS = ('attack', 'a ttack', '1', 'anomaly', 'true')


def load_trace(path, nrows=None):
    """Load a SWaT-style CSV/XLSX; returns (X [T,p] float, labels [T] {0,1}, cols)."""
    read = pd.read_excel if str(path).lower().endswith(('.xlsx', '.xls')) else pd.read_csv
    df = read(path, nrows=nrows)
    df.columns = [str(c).strip() for c in df.columns]
    label_col = next((c for c in df.columns if c.lower().strip() in LABEL_ALIASES), None)
    y = np.zeros(len(df), dtype=int)
    if label_col is not None:
        vals = df[label_col].astype(str).str.strip().str.lower()
        y = vals.isin(ATTACK_TOKENS).astype(int).to_numpy()
        df = df.drop(columns=[label_col])
    num = df.select_dtypes(include=[np.number])
    # drop constant/actuator-like binary columns (uninformative for residuals)
    keep = [c for c in num.columns if num[c].nunique() > 5]
    return num[keep], y


def downsample(X, y, k):
    if k <= 1:
        return X, y
    T = (len(X) // k) * k
    Xd = np.median(X[:T].reshape(-1, k, X.shape[1]), axis=1)
    yd = (y[:T].reshape(-1, k).max(axis=1))
    return Xd, yd


class RidgeForecaster:
    """Multi-output one-step ridge forecaster on n_lags lags (deterministic)."""

    def __init__(self, n_lags=4, alpha=1.0):
        self.L, self.alpha = n_lags, alpha

    def _design(self, X):
        L = self.L
        Z = np.hstack([X[L - l - 1:len(X) - l - 1] for l in range(L)])
        return Z, X[L:]

    def fit(self, X):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu_) / self.sd_
        Z, Y = self._design(Xs)
        self.model_ = Ridge(alpha=self.alpha).fit(Z, Y)
        return self

    def residuals(self, X):
        Xs = (X - self.mu_) / self.sd_
        Z, Y = self._design(Xs)
        r = Y - self.model_.predict(Z)
        return r  # aligned to timestamps L..end


def contiguous_segments(y):
    segs, start = [], None
    for i, v in enumerate(y):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i)); start = None
    if start is not None:
        segs.append((start, len(y)))
    return segs


def windows_from(r, idx_start, count, W_L, stride=None):
    """`count` non-overlapping W_L windows beginning at residual index idx_start."""
    stride = stride or W_L
    out, i = [], idx_start
    while len(out) < count and i + W_L <= len(r):
        out.append(r[i:i + W_L].flatten()); i += stride
    return np.array(out), i


def run_protocol(normal_path, attack_path, ds=5, W_L=10, d_pca=12,
                 train_frac=0.6, n_lags=4, seed=42, nrows=None, verbose=True,
                 q_scale=1.0):
    """q_scale < 1 applies bandwidth-controlled input rescaling to the
    quantum-kernel path only (design and scoring), per the concentration-aware
    study; classical detectors are unaffected."""
    rng = np.random.RandomState(seed)
    Dn, _ = load_trace(normal_path, nrows=nrows)
    Da, ya = load_trace(attack_path, nrows=nrows)
    # channels must be selected identically for both traces: name intersection
    cols = [c for c in Dn.columns if c in set(Da.columns)]
    to_x = lambda D: np.nan_to_num(D[cols].to_numpy(dtype=float),
                                   nan=0.0, posinf=0.0, neginf=0.0)
    Xn, Xa = to_x(Dn), to_x(Da)
    Xn, _ = downsample(Xn, np.zeros(len(Xn), int), ds)
    Xa, ya = downsample(Xa, ya, ds)
    if verbose:
        print(f"  loaded: normal {Xn.shape}, attack {Xa.shape}, "
              f"{len(cols)} common channels, {ya.sum()} attack samples")

    n_train = int(train_frac * len(Xn))
    fc = RidgeForecaster(n_lags=n_lags).fit(Xn[:n_train])
    r_nom = fc.residuals(Xn[n_train:])
    r_atk_full = fc.residuals(Xa)
    y_atk = ya[n_lags:]

    # normal windows: ref(20) -> cal(30) -> test(30), chronological, disjoint
    W_ref, i1 = windows_from(r_nom, 0, 20, W_L)
    W_cal, i2 = windows_from(r_nom, i1, 30, W_L)
    W_tst, _ = windows_from(r_nom, i2, 30, W_L)
    if min(len(W_ref), len(W_cal), len(W_tst)) < 20:
        raise RuntimeError("normal trace too short for the protocol at this ds/W_L")

    # one centered window per labeled attack scenario
    segs = contiguous_segments(y_atk)
    W_atk, seg_ids = [], []
    for si, (a, b) in enumerate(segs):
        if b - a < W_L:
            continue
        c = (a + b) // 2 - W_L // 2
        W_atk.append(r_atk_full[c:c + W_L].flatten()); seg_ids.append(si)
    W_atk = np.array(W_atk)
    if verbose:
        print(f"  windows: ref {len(W_ref)}, cal {len(W_cal)}, test {len(W_tst)}, "
              f"attack scenarios usable {len(W_atk)}/{len(segs)}")
    if len(segs) < 5:
        raise RuntimeError(
            f"only {len(segs)} contiguous attack segment(s) found -- the attack "
            "trace appears to be label-filtered (attack rows only, scenarios "
            "spliced together). Use the ORIGINAL interleaved trace with normal "
            "and attack rows in time order (e.g., SWaT_Dataset_Attack_v0 / "
            "merged.csv); per-scenario metrics are meaningless otherwise.")
    if len(W_atk) < 5:
        raise RuntimeError(
            f"only {len(W_atk)} attack scenarios long enough for W_L={W_L} at "
            f"this downsampling -- reduce --ds or --wl.")

    # standardize by reference stats; PCA on reference
    mu_w, sd_w = W_ref.mean(0), W_ref.std(0) + 1e-8
    std = lambda W: (W - mu_w) / sd_w
    mu, V = pca_fit(std(W_ref), d_pca)
    proj = lambda W: (std(W) - mu) @ V
    Xr, Xc, Xt, Xk = proj(W_ref), proj(W_cal), proj(W_tst), proj(W_atk)

    # quantum kernel: theta designed on ref vs first half of attack windows
    qc = QuantumCircuit(cfg.N_QUBITS, cfg.CIRCUIT_DEPTH)
    theta, _ = design_quantum_kernel(qc, Xr * q_scale,
                                     Xk[:max(2, len(Xk) // 2)] * q_scale, seed)
    qcache = QKernelCache(qc, theta)
    S_ref = qcache.sv(Xr * q_scale)
    K_q = gram_from_sv(S_ref); np.fill_diagonal(K_q, 1.0)
    gamma = rbf_bandwidth(Xr, Xk[:max(2, len(Xk) // 2)])
    Kr = sk_rbf(Xr, Xr, gamma); Kl = sk_lap(Xr, Xr, gamma)

    def kscore(K_ref, K_cross):
        m = len(K_ref)
        mr = (np.sum(K_ref) - np.trace(K_ref)) / (m * (m - 1))
        return mr + 1.0 - 2.0 * K_cross.mean(axis=0)

    oc_t, _, _ = fit_tuned_ocsvm(Xr, Xk[:max(2, len(Xk) // 2)])
    if_t, _, _ = fit_tuned_iforest(Xr, Xk[:max(2, len(Xk) // 2)], seed=42)
    ae = WindowAutoencoder(seed=0).fit(Xr)
    oc_d = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale').fit(Xr)

    fns = {
        'Q-Direct': lambda X: kscore(K_q, gram_from_sv(S_ref, qcache.sv(X * q_scale))),
        'RBF':      lambda X: kscore(Kr, sk_rbf(Xr, X, gamma)),
        'Laplacian':lambda X: kscore(Kl, sk_lap(Xr, X, gamma)),
        'OC-SVM^t': lambda X: -oc_t.decision_function(X),
        'IForest^t':lambda X: -if_t.decision_function(X),
        'AE':       ae.score,
        'OC-SVM':   lambda X: -oc_d.decision_function(X),
        'Res-norm': lambda X: np.linalg.norm(X, axis=1),
    }

    results, per_scenario = {}, {}
    y = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xk))])
    for name, fn in fns.items():
        tau = np.percentile(fn(Xc), 95)
        s_neg, s_pos = fn(Xt), fn(Xk)
        sc = np.concatenate([s_neg, s_pos])
        results[name] = dict(
            auc=float(roc_auc_score(y, sc)),
            f1=float(f1_score(y, (sc > tau).astype(int), zero_division=0)),
            fpr=float(np.mean(s_neg > tau)),
            tpr=float(np.mean(s_pos > tau)))
        per_scenario[name] = [(seg_ids[i], bool(s_pos[i] > tau)) for i in range(len(Xk))]
        # withheld-half metrics: scenarios beyond the design slice Xk[:max(2, len(Xk)//2)]
        nh = max(2, len(Xk) // 2)
        y_wh = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xk) - nh)])
        sc_wh = np.concatenate([s_neg, s_pos[nh:]])
        results[name].update(
            auc_wh=float(roc_auc_score(y_wh, sc_wh)),
            f1_wh=float(f1_score(y_wh, (sc_wh > tau).astype(int), zero_division=0)),
            tpr_wh=float(np.mean(s_pos[nh:] > tau)))

    # concentration diagnostic
    Krn = gram_from_sv(S_ref, qcache.sv(Xt * q_scale)).ravel()
    Kra = gram_from_sv(S_ref, qcache.sv(Xk * q_scale)).ravel()
    Krr = K_q[np.triu_indices(len(K_q), 1)]
    conc = dict(ref_ref=(float(Krr.mean()), float(Krr.std())),
                ref_nom=(float(Krn.mean()), float(Krn.std())),
                ref_atk=(float(Kra.mean()), float(Kra.std())))
    return dict(results=results, per_scenario=per_scenario, concentration=conc,
                n_scenarios=len(segs), n_used=len(W_atk), channels=len(cols),
                gamma=float(gamma), seed=seed, q_scale=q_scale)
