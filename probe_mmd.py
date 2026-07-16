"""Encoding-scale probe: batch MMD_u^2 and held-out AUC vs input scale c.

Because the encoding angles are theta_{l,i,r} * xi_j, rescaling the inputs by c is
algebraically equivalent to evaluating the kernel at c*theta_dagger: the probe samples
the same parameter space that the pooled SPSA design searched.

Reports BOTH:
  (i)  the POOLED design objective (what Algorithm 1 actually maximizes), and
  (ii) the per-family statistic and held-out AUC for the two probed cases.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import roc_auc_score
from asqdt.scripts.run_v2_revision import load_or_build_mp
from asqdt.scripts.run_v2_honest import fresh_noms, kscore
from asqdt.evaluation.fastq import gram_from_sv, mmd_u, batch_statevectors

MP = load_or_build_mp(set())
SCALES = (1.0, 0.1)

def mmd_at(mp, Xa, c):
    Sn = batch_statevectors(mp['qc'], mp['Xn_p'] * c, mp['theta'])
    Sa = batch_statevectors(mp['qc'], Xa * c, mp['theta'])
    return mmd_u(gram_from_sv(Sn), gram_from_sv(Sa), gram_from_sv(Sn, Sa))

print("== POOLED design objective (the quantity Algorithm 1 maximizes) ==")
print(f"{'bench':<8}{'seed':>5}" + "".join(f"{'MMD@c='+str(c):>16}" for c in SCALES))
for bname in ('CNC', 'CSTR'):
    for seed in (42, 123, 256, 789, 1024):
        mp = MP[bname][seed]
        vals = [mmd_at(mp, mp['Xa_pool'], c) for c in SCALES]
        print(f"{bname:<8}{seed:>5}" + "".join(f"{v:>16.4f}" for v in vals))

print("\n== Per-family statistic and held-out AUC (seed 42) ==")
print(f"{'bench/family':<22}{'c':>6}{'batch MMD_u^2':>16}{'held-out AUC':>15}")
for bname, fam in (('CNC', 'command'), ('CSTR', 'cstr_thermal')):
    mp = MP[bname][42]; d = mp['data']
    F  = (fresh_noms(bname, 42, d, 30) - mp['mu']) @ mp['V']
    Xa = (np.nan_to_num(d['atk'][fam]['windows']) - mp['mu']) @ mp['V']
    y  = np.concatenate([np.zeros(len(F)), np.ones(len(Xa))])
    for c in SCALES:
        m2 = mmd_at(mp, Xa, c)
        Sn = batch_statevectors(mp['qc'], mp['Xn_p'] * c, mp['theta'])
        K = gram_from_sv(Sn); np.fill_diagonal(K, 1.0)
        sc = lambda X: kscore(K, gram_from_sv(Sn, batch_statevectors(mp['qc'], X * c, mp['theta'])))
        auc = roc_auc_score(y, np.concatenate([sc(F), sc(Xa)]))
        print(f"{bname+'/'+fam:<22}{c:>6}{m2:>16.4f}{auc:>15.2f}")
