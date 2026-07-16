#!/usr/bin/env python3
"""
Follow-up study: concentration-aware quantum feature maps + n_q scaling,
evaluated EXCLUSIVELY under the corrected held-out protocol (Supp. L).

NOT part of the locked v2 submission -- this implements the first and fifth
items of the paper's future-work list as the seed of the follow-up paper.

Stages:
  CA_CNC / CA_CSTR   Concentration-aware design variants, 5 seeds:
      base     current design (reference)
      scale50  inputs rescaled by c=0.50 before encoding, theta redesigned
      scale25  c=0.25 redesign
      scale10  c=0.10 redesign
      ktarget  SPSA objective = MMD - 5*(kbar_nom - 0.5)^2  (bandwidth-
               targeted: pushes the mean off-diagonal reference similarity
               toward 0.5 instead of letting it collapse)
      ridge    SPSA objective = MMD - 0.05*||theta||^2 (magnitude-regularized)
    All variants are fixed a priori (no per-seed selection -> no test leakage).
    Reported per variant: kbar bands (ref-ref / ref-fresh-nominal / ref-attack),
    held-out AUC and F1 per attack family, batch MMD.
  NQ_SCALING   n_q in {2,3,4,6,8} at depth 3 (PCA d = 3*n_q), base design,
    CNC + CSTR: kbar and band widths vs n_q, held-out AUC vs n_q. Tests the
    exponential-concentration prediction (Thanasilp et al. 2024) in-setting.

Usage: python -m asqdt.scripts.run_concentration_aware [CA_CNC] [CA_CSTR]
       [NQ_SCALING] [--quick]   (--quick: seed 42 only, for smoke tests)
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics import roc_auc_score, f1_score

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import load_or_build_mp, RESULTS, save_stage
from asqdt.scripts.run_v2_honest import fresh_noms, ensure_tank_stats, kscore
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.evaluation.fastq import (QKernelCache, gram_from_sv, mmd_u, pca_fit,
                                    batch_statevectors)

ALPHA = 0.05


def design_theta(qc, Xn, Xa, seed, scale=1.0, k_target=None, lam_k=5.0,
                 lam_ridge=0.0, iters=None, lr=None):
    """SPSA design with optional input scaling / kbar-target / theta-ridge.

    Replicates the pipeline's SPSA loop (init U[0.5,1.5], best-iterate,
    weight decay 0.02) with the objective
        J(theta) = MMD_u^2 - lam_k*(kbar_nom - k_target)^2 - lam_ridge*||theta||^2/|theta| .
    """
    iters = iters or cfg.MAX_ITERS
    lr = lr or cfg.KERNEL_LR
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0.5, 1.5, size=(qc.depth, qc.nq, 2))
    Xn_s, Xa_s = Xn * scale, Xa * scale

    def objective(t):
        Sn = batch_statevectors(qc, Xn_s, t)
        Sa = batch_statevectors(qc, Xa_s, t)
        Knn = gram_from_sv(Sn); np.fill_diagonal(Knn, 1.0)
        Kaa = gram_from_sv(Sa); np.fill_diagonal(Kaa, 1.0)
        J = mmd_u(Knn, Kaa, gram_from_sv(Sn, Sa))
        if k_target is not None:
            m = len(Knn)
            kbar = (np.sum(Knn) - np.trace(Knn)) / (m * (m - 1))
            J -= lam_k * (kbar - k_target) ** 2
        if lam_ridge > 0:
            J -= lam_ridge * np.mean(t ** 2)
        return J

    best_J, best_t = objective(theta), theta.copy()
    eps = 0.1
    for _ in range(iters):
        delta = rng.choice([-1, 1], size=theta.shape)
        Jp = objective(theta + eps * delta)
        Jm = objective(theta - eps * delta)
        g = (Jp - Jm) / (2 * eps) * delta
        theta = theta + lr * (g - 0.02 * theta)
        J = objective(theta)
        if J > best_J:
            best_J, best_t = J, theta.copy()
    return best_t


VARIANTS = [
    ('base',    dict()),
    ('scale50', dict(scale=0.50)),
    ('scale25', dict(scale=0.25)),
    ('scale10', dict(scale=0.10)),
    ('ktarget', dict(k_target=0.5)),
    ('ridge',   dict(lam_ridge=0.05)),
]


def eval_variant(mp, bname, seed, d, theta, scale):
    """Held-out metrics + kbar bands for one designed kernel."""
    qc = mp['qc']
    F = (fresh_noms(bname, seed, d, 60) - mp['mu']) @ mp['V']
    Xcal, Xtst = F[:30] * scale, F[30:] * scale
    Xr = mp['Xn_p'] * scale
    Sr = batch_statevectors(qc, Xr, theta)
    K = gram_from_sv(Sr); np.fill_diagonal(K, 1.0)
    m = len(K)
    kbar = (np.sum(K) - np.trace(K)) / (m * (m - 1))
    sc = lambda X: kscore(K, gram_from_sv(Sr, batch_statevectors(qc, X, theta)))
    tau = np.percentile(sc(Xcal), 100 * (1 - ALPHA))
    s_neg = sc(Xtst)
    out = {'kbar': float(kbar),
           'band_nom': (float(np.mean(gram_from_sv(Sr, batch_statevectors(qc, Xtst, theta)))), ),
           'fam': {}}
    for a in d['atk']:
        Xa = ((np.nan_to_num(d['atk'][a]['windows']) - mp['mu']) @ mp['V']) * scale
        s_pos = sc(Xa)
        y = np.concatenate([np.zeros(30), np.ones(len(Xa))])
        s = np.concatenate([s_neg, s_pos])
        Sa = batch_statevectors(qc, Xa, theta)
        Kaa = gram_from_sv(Sa); np.fill_diagonal(Kaa, 1.0)
        out['fam'][a] = dict(
            auc=float(roc_auc_score(y, s)),
            f1=float(f1_score(y, (s > tau).astype(int), zero_division=0)),
            mmd=float(mmd_u(K, Kaa, gram_from_sv(Sr, Sa))))
    return out


def stage_CA(MP, bname, seeds):
    print("\n" + "=" * 76)
    print(f" CONCENTRATION-AWARE DESIGN VARIANTS ({bname}; held-out protocol)")
    print("=" * 76)
    fams = list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
    agg = {v: {a: {'auc': [], 'f1': [], 'mmd': []} for a in fams} for v, _ in VARIANTS}
    kbars = {v: [] for v, _ in VARIANTS}
    for seed in seeds:
        mp = MP[bname][seed]; d = mp['data']
        Xa_pool = mp['Xa_pool'][:len(mp['Xn_p'])]
        for vname, kw in VARIANTS:
            scale = kw.get('scale', 1.0)
            theta = design_theta(mp['qc'], mp['Xn_p'] * scale, Xa_pool * scale,
                                 seed, scale=1.0,  # inputs pre-scaled above
                                 k_target=kw.get('k_target'),
                                 lam_ridge=kw.get('lam_ridge', 0.0))
            r = eval_variant(mp, bname, seed, d, theta, scale)
            kbars[vname].append(r['kbar'])
            for a in fams:
                for k in ('auc', 'f1', 'mmd'):
                    agg[vname][a][k].append(r['fam'][a][k])
            aucs = " ".join(f"{r['fam'][a]['auc']:.2f}" for a in fams)
            print(f"  {bname} seed={seed} {vname:8s} kbar={r['kbar']:.3f} "
                  f"AUC[{'/'.join(fams)}]= {aucs}", flush=True)
    print(f"\n  LaTeX rows ({bname}; variant & kbar & per-family held-out AUC):")
    out = {}
    for vname, _ in VARIANTS:
        cells = " & ".join(
            f"{np.mean(agg[vname][a]['auc']):.2f}$\\pm${np.std(agg[vname][a]['auc']):.2f}"
            for a in fams)
        kb = f"{np.mean(kbars[vname]):.2f}$\\pm${np.std(kbars[vname]):.2f}"
        print(f"    {vname} & {kb} & {cells} \\\\")
        out[vname] = dict(kbar=kb, fam={a: {k: (float(np.mean(agg[vname][a][k])),
                                                float(np.std(agg[vname][a][k])))
                                           for k in ('auc', 'f1', 'mmd')} for a in fams})
    save_stage(f'concaware_{bname}', out)


def stage_NQ(MP, seeds):
    print("\n" + "=" * 76)
    print(" n_q SCALING: concentration and held-out AUC vs qubit count (depth 3)")
    print("=" * 76)
    out = {}
    for bname in ('CNC', 'CSTR'):
        out[bname] = {}
        for nq in (2, 3, 4, 6, 8):
            d_pca = 3 * nq
            rows = {'kbar': [], 'band': [], 'auc': []}
            for seed in seeds:
                mp = MP[bname][seed]; d = mp['data']
                # refit PCA at the matching dimension
                Xn_raw = d['nom']
                mu, V = pca_fit(Xn_raw, d_pca)
                Xr = (Xn_raw - mu) @ V
                Xa_pool = (np.nan_to_num(
                    np.vstack([d['atk'][a]['windows'] for a in d['atk']])[:len(Xr)]
                    ) - mu) @ V
                qc = QuantumCircuit(nq, cfg.CIRCUIT_DEPTH)
                theta = design_theta(qc, Xr, Xa_pool, seed)
                Sr = batch_statevectors(qc, Xr, theta)
                K = gram_from_sv(Sr); np.fill_diagonal(K, 1.0)
                m = len(K)
                kb = (np.sum(K) - np.trace(K)) / (m * (m - 1))
                offd = K[np.triu_indices(m, 1)]
                F = (fresh_noms(bname, seed, d, 60) - mu) @ V
                sc = lambda X: kscore(K, gram_from_sv(Sr, batch_statevectors(qc, X, theta)))
                s_neg = sc(F[30:])
                aucs = []
                for a in d['atk']:
                    Xa = (np.nan_to_num(d['atk'][a]['windows']) - mu) @ V
                    y = np.concatenate([np.zeros(30), np.ones(len(Xa))])
                    aucs.append(roc_auc_score(y, np.concatenate([s_neg, sc(Xa)])))
                rows['kbar'].append(kb); rows['band'].append(float(offd.std()))
                rows['auc'].append(float(np.mean(aucs)))
            out[bname][nq] = {k: (float(np.mean(v)), float(np.std(v)))
                              for k, v in rows.items()}
            r = out[bname][nq]
            print(f"  {bname} n_q={nq} (d={d_pca}): kbar={r['kbar'][0]:.3f}"
                  f"±{r['kbar'][1]:.3f}  band_std={r['band'][0]:.3f}"
                  f"  mean held-out AUC={r['auc'][0]:.2f}±{r['auc'][1]:.2f}", flush=True)
    save_stage('nq_scaling', out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    quick = '--quick' in sys.argv
    seeds = [42] if quick else cfg.SEEDS
    stages = args or ['CA_CNC']
    MP = load_or_build_mp(set())
    ensure_tank_stats(MP)
    if 'CA_CNC' in stages: stage_CA(MP, 'CNC', seeds)
    if 'CA_CSTR' in stages: stage_CA(MP, 'CSTR', seeds)
    if 'NQ_SCALING' in stages: stage_NQ(MP, seeds)
    print("\nDONE:", stages, "(quick)" if quick else "")


if __name__ == '__main__':
    main()
