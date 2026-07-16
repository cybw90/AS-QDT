"""
Adaptive-attack stress test (reviewers R3.3 / R4.2; Supplementary Section O).

White-box adversary against the FROZEN final design (s_dag, theta_dag):
optimizes the attack signal a in R^{H x q} to minimize the batch MMD U-statistic
of the resulting residual windows against the nominal reference, subject to the
attack-set constraints of A_{rho,eps} (impact >= rho, legacy score <= eps,
energy <= E_max via projection; state/input feasibility via the clamped plant).

Optimizer: augmented-Lagrangian-STYLE quadratic penalty method with SPSA
gradients in signal space (2 rollout batches per iteration, dimension-free)
and multiplicative penalty ramping on violated constraints. NOTE: the paper's
Supplementary C describes L-BFGS-B on a differentiable simulator; this NumPy
simulator is not autodifferentiable, so SPSA is used. Update the supplement
wording accordingly (flagged in the run report).

Episode-level noise is held to the SAME fixed episode seeds used by the
benchmark generators, so the objective is deterministic in a and the resulting
fixed attack is evaluated over repeated noise episodes (Remark 2 split).
"""
import numpy as np
from .evaluation.fastq import mmd_u, gram_from_sv


def _project_energy(a, E_max):
    e = np.sum(a ** 2)
    if e > E_max:
        a = a * np.sqrt(E_max / e)
    return a


class AdaptiveAttacker:
    """Benchmark-agnostic adaptive attacker.

    rollout_fn(a_seq, ep) -> dict(window_raw, impact, legacy)
        simulates episode `ep` (fixed noise seed) under fixed attack a_seq and
        returns the standardized raw window, realized impact and legacy score.
    """

    def __init__(self, rollout_fn, n_eval_ep, params, pca_mu, pca_V,
                 qcache, S_nom, K_nom):
        self.rollout = rollout_fn
        self.n_ep = n_eval_ep
        self.rho = params['rho']
        self.eps = params['epsilon']
        self.E_max = params['E_max']
        self.pca_mu, self.pca_V = pca_mu, pca_V
        self.qc = qcache                      # QKernelCache at theta_dag
        self.S_nom = S_nom                    # cached nominal statevectors
        self.K_nom = K_nom                    # cached nominal Gram

    # ── evaluation of a fixed attack over the eval episodes ─────────
    def evaluate(self, a_seq):
        wins, imps, legs = [], [], []
        for ep in range(self.n_ep):
            out = self.rollout(a_seq, ep)
            wins.append(out['window_raw'])
            imps.append(out['impact'])
            legs.append(out['legacy'])
        Xa = (np.nan_to_num(np.array(wins)) - self.pca_mu) @ self.pca_V
        Sa = self.qc.sv(Xa)
        Kaa = gram_from_sv(Sa)
        np.fill_diagonal(Kaa, 1.0)
        mmd = mmd_u(self.K_nom, Kaa, gram_from_sv(self.S_nom, Sa))
        in_set = sum(1 for i in range(self.n_ep)
                     if imps[i] >= self.rho and legs[i] <= self.eps)
        return dict(mmd=mmd, impacts=np.array(imps), legacies=np.array(legs),
                    in_set=in_set, Xa=Xa)

    def _penalized(self, a_seq, lam_I, lam_D):
        ev = self.evaluate(a_seq)
        vI = np.maximum(0.0, self.rho - ev['impacts']) / max(self.rho, 1e-9)
        vD = np.maximum(0.0, ev['legacies'] - self.eps) / max(self.eps, 1e-9)
        pen = lam_I * np.mean(vI ** 2) + lam_D * np.mean(vD ** 2)
        return ev['mmd'] + pen, ev

    # ── SPSA penalty-method optimization ─────────────────────────────
    def optimize(self, a_init, n_iters=80, seed=0, spsa_eps_frac=0.05,
                 lr_frac=0.08, lam0=0.5, ramp=2.0, ramp_every=10, verbose=False):
        rng = np.random.RandomState(seed)
        scale = max(np.max(np.abs(a_init)), 1e-3)
        eps = spsa_eps_frac * scale
        a = _project_energy(a_init.copy(), self.E_max)
        lam_I = lam_D = lam0

        best = None   # (mmd, a, ev) among fully in-set iterates
        best_any = None

        for it in range(n_iters):
            delta = rng.choice([-1.0, 1.0], size=a.shape)
            fp, _ = self._penalized(_project_energy(a + eps * delta, self.E_max),
                                    lam_I, lam_D)
            fm, _ = self._penalized(_project_energy(a - eps * delta, self.E_max),
                                    lam_I, lam_D)
            g = (fp - fm) / (2 * eps) * delta
            lr = lr_frac * scale / (1 + it / 20.0)
            a = _project_energy(a - lr * g, self.E_max)

            f, ev = self._penalized(a, lam_I, lam_D)
            feasible = ev['in_set'] == self.n_ep
            if feasible and (best is None or ev['mmd'] < best[0]):
                best = (ev['mmd'], a.copy(), ev)
            if best_any is None or (ev['in_set'], -ev['mmd']) > (best_any[0], -best_any[1]):
                best_any = (ev['in_set'], ev['mmd'], a.copy(), ev)
            if (it + 1) % ramp_every == 0:
                if np.any(ev['impacts'] < self.rho):
                    lam_I *= ramp
                if np.any(ev['legacies'] > self.eps):
                    lam_D *= ramp
            if verbose and (it + 1) % 20 == 0:
                print(f"      it={it+1:3d} mmd={ev['mmd']:+.5f} "
                      f"in_set={ev['in_set']}/{self.n_ep} lam=({lam_I:.1f},{lam_D:.1f})")

        if best is not None:
            return best[1], best[2]
        # fall back to the most-feasible iterate (report its in-set rate honestly)
        return best_any[2], best_any[3]
