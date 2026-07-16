"""
Evaluation: metrics, classical baselines, and complete experiment runner.

Baselines:
  B1: Chi-square / CUSUM residual threshold
  B2: DT + classical one-class SVM
  B3: DT + RBF kernel MMD (optimized bandwidth)
  B4: DT + Laplacian kernel MMD
  B5: All sensors (no selection)
  B6: Random sensor selection
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM
from ..quantum.kernel import rbf_kernel, laplacian_kernel
from ..detection.mmd import mmd_from_embeddings, mmd_squared, permutation_threshold
from ..quantum.surrogate import NystromSurrogate
from .. import config as cfg


# ── Metrics ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_scores, y_pred=None, threshold=None):
    """Compute detection metrics."""
    if y_pred is None:
        if threshold is None:
            threshold = np.median(y_scores)
        y_pred = (y_scores > threshold).astype(int)
    
    results = {}
    results['f1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        results['auroc'] = roc_auc_score(y_true, y_scores)
    except ValueError:
        results['auroc'] = 0.5
    results['fpr'] = np.mean(y_pred[y_true == 0])
    results['fnr'] = 1.0 - np.mean(y_pred[y_true == 1]) if np.sum(y_true == 1) > 0 else 0.0
    results['tpr'] = 1.0 - results['fnr']
    return results


def detection_delay(y_scores, threshold, attack_start_idx=0):
    """Samples from attack start until first detection."""
    for i in range(attack_start_idx, len(y_scores)):
        if y_scores[i] > threshold:
            return i - attack_start_idx
    return len(y_scores) - attack_start_idx


# ── Baselines ──────────────────────────────────────────────────────

class ChiSquareDetector:
    """B1: Chi-square residual threshold detector."""
    
    def __init__(self):
        self.threshold = None
    
    def fit(self, residual_windows):
        norms = np.sum(residual_windows ** 2, axis=1)
        self.threshold = np.percentile(norms, 100 * (1 - cfg.DELTA))
    
    def score(self, residual_windows):
        return np.sum(residual_windows ** 2, axis=1)
    
    def predict(self, residual_windows):
        scores = self.score(residual_windows)
        return (scores > self.threshold).astype(int)


class OneClassSVMDetector:
    """B2: DT residuals + one-class SVM."""
    
    def __init__(self, kernel='rbf', nu=0.1):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma='scale')
    
    def fit(self, nominal_windows):
        self.model.fit(nominal_windows)
    
    def score(self, windows):
        return -self.model.decision_function(windows)
    
    def predict(self, windows):
        return (self.model.predict(windows) == -1).astype(int)


class ClassicalKernelMMD:
    """B3/B4: DT + classical kernel MMD detector."""
    
    def __init__(self, kernel_type='rbf', gamma=None):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.surrogate = NystromSurrogate(rank=cfg.NYSTROM_RANK)
        self.phi_nom = None
        self.threshold = None
    
    def fit(self, nominal_windows):
        if self.gamma is None:
            self.gamma = 1.0 / nominal_windows.shape[1]
        
        kernel_fn = lambda X, Y: (rbf_kernel(X, Y, self.gamma) if self.kernel_type == 'rbf'
                                   else laplacian_kernel(X, Y, self.gamma))
        
        n_landmarks = min(cfg.NYSTROM_RANK, len(nominal_windows))
        rng = np.random.RandomState(0)
        landmarks = nominal_windows[rng.choice(len(nominal_windows), n_landmarks, replace=False)]
        self.surrogate.fit(landmarks, kernel_fn)
        self.phi_nom = self.surrogate.embed(nominal_windows, kernel_fn)
        self.kernel_fn = kernel_fn
        self.threshold = permutation_threshold(self.phi_nom)
    
    def score_windows(self, test_windows):
        phi_test = self.surrogate.embed(test_windows, self.kernel_fn)
        # Compute MMD² between nominal and test
        return mmd_from_embeddings(self.phi_nom, phi_test)
    
    def score_per_window(self, test_windows, batch_size=10):
        """Score in sliding batches for per-window detection."""
        scores = []
        phi_test = self.surrogate.embed(test_windows, self.kernel_fn)
        for i in range(0, len(phi_test), batch_size):
            batch = phi_test[i:i+batch_size]
            if len(batch) < 3:
                scores.extend([0.0] * len(batch))
                continue
            s = mmd_from_embeddings(self.phi_nom[:batch_size], batch)
            scores.extend([s] * len(batch))
        return np.array(scores)


def optimize_rbf_bandwidth(X_nom, X_atk, gammas=None):
    """Optimize RBF bandwidth for maximum MMD² (Sutherland et al. 2017 style)."""
    if gammas is None:
        median_dist = np.median(np.linalg.norm(X_nom[:50, None] - X_nom[None, :50], axis=2))
        gammas = [0.1 / median_dist**2, 0.5 / median_dist**2, 
                  1.0 / median_dist**2, 2.0 / median_dist**2, 5.0 / median_dist**2]
    
    best_gamma = gammas[0]
    best_mmd = -np.inf
    
    for g in gammas:
        K_XX = rbf_kernel(X_nom[:50], X_nom[:50], g)
        K_YY = rbf_kernel(X_atk[:50], X_atk[:50], g)
        K_XY = rbf_kernel(X_nom[:50], X_atk[:50], g)
        val = mmd_squared(K_XX, K_YY, K_XY)
        if val > best_mmd:
            best_mmd = val
            best_gamma = g
    
    return best_gamma


def run_baseline_evaluation(nom_windows, attack_data, verbose=True):
    """Run all baselines and return comparative metrics."""
    results = {}
    
    # Prepare labels and data
    n_nom = min(len(nom_windows), 200)
    nom_sub = nom_windows[:n_nom]
    
    for atk_name, atk_info in attack_data.items():
        atk_win = atk_info['windows']
        n_atk = min(len(atk_win), 200)
        atk_sub = atk_win[:n_atk]
        
        y_true = np.concatenate([np.zeros(n_nom), np.ones(n_atk)])
        all_windows = np.vstack([nom_sub, atk_sub])
        
        atk_results = {}
        
        # B1: Chi-square
        det_chi = ChiSquareDetector()
        det_chi.fit(nom_sub)
        scores_chi = det_chi.score(all_windows)
        pred_chi = det_chi.predict(all_windows)
        atk_results['chi_square'] = compute_metrics(y_true, scores_chi, pred_chi)
        
        # B2: One-class SVM
        det_svm = OneClassSVMDetector()
        det_svm.fit(nom_sub)
        scores_svm = det_svm.score(all_windows)
        pred_svm = det_svm.predict(all_windows)
        atk_results['ocsvm'] = compute_metrics(y_true, scores_svm, pred_svm)
        
        # B3: Optimized RBF-MMD
        best_gamma = optimize_rbf_bandwidth(nom_sub, atk_sub)
        det_rbf = ClassicalKernelMMD('rbf', best_gamma)
        det_rbf.fit(nom_sub)
        scores_rbf = det_rbf.score_per_window(all_windows)
        tau_rbf = det_rbf.threshold
        pred_rbf = (scores_rbf > tau_rbf).astype(int)
        atk_results['rbf_mmd'] = compute_metrics(y_true, scores_rbf, pred_rbf)
        
        # B4: Laplacian MMD
        det_lap = ClassicalKernelMMD('laplacian', best_gamma)
        det_lap.fit(nom_sub)
        scores_lap = det_lap.score_per_window(all_windows)
        tau_lap = det_lap.threshold
        pred_lap = (scores_lap > tau_lap).astype(int)
        atk_results['laplacian_mmd'] = compute_metrics(y_true, scores_lap, pred_lap)
        
        results[atk_name] = atk_results
        
        if verbose:
            print(f"\n  Attack: {atk_name}")
            for method, m in atk_results.items():
                print(f"    {method:15s}: F1={m['f1']:.3f}  AUROC={m['auroc']:.3f}  "
                      f"FPR={m['fpr']:.3f}  TPR={m['tpr']:.3f}")
    
    return results
