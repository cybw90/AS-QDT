#!/usr/bin/env python3
"""
export_hw_inputs.py — prepare inputs for the IBM Quantum concentration check.

Run from the repo root AFTER applying sign_fix.patch (the export is only
meaningful under the canonical sign-deterministic pipeline):

    python -m asqdt.scripts.export_hw_inputs

Regenerates the seed-42 CNC canonical quantum design (identical to
run_tifs_v2: initial theta + 6 SPSA iterations, best iterate retained),
selects 25 window pairs per concentration band (ref-ref, ref-fresh-nominal,
ref-attack[sinusoidal]), and writes hw_inputs.json containing:

  - theta          : the designed 3x4x2 input-scaling parameters
  - vectors        : the PCA-projected 12-dim feature vectors involved
  - pairs          : index pairs per band
  - ref_fidelity   : exact-statevector k_theta for every pair at c=1.0 and c=0.1

The reference fidelities let the hardware script self-check its circuit
construction before spending QPU time, and give the exact-simulation column
of the Supplementary~R table.
"""
import sys, os, json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt import config as cfg
from asqdt.cpms.data import generate_dataset
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.detection.mmd import mmd_squared
from asqdt.scripts.run_tifs_v2 import pca_fit
from asqdt.scripts.run_v2_honest import fresh_noms

SEED, NQ, DEPTH, N_PAIRS = 42, 4, 3, 25
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'hw_inputs.json')


def design_theta(Xn_p, Xa_pool, qc, seed):
    """Exact replica of the run_tifs_v2 canonical design loop."""
    rng_used = np.random.RandomState(seed)
    theta = qc.initial_params(rng_used)
    best_m, best_t = -np.inf, theta.copy()

    def obj(th):
        return mmd_squared(qc.gram_matrix(Xn_p, th), qc.gram_matrix(Xa_pool, th),
                           qc.cross_gram(Xn_p, Xa_pool, th))

    for _ in range(cfg.MAX_ITERS):
        m_val = obj(theta)
        if m_val > best_m:
            best_m, best_t = m_val, theta.copy()
        delta = rng_used.choice([-1, 1], size=theta.shape)
        g = (obj(theta + 0.1 * delta) - obj(theta - 0.1 * delta)) / 0.2 * delta
        theta = theta + cfg.KERNEL_LR * (g - 0.02 * theta)
    return best_t


def main():
    Xn, atk, _, nom_mean, nom_std = generate_dataset(
        mismatch_frac=0.05, seed=SEED, W_L=10, E_max=2.0, legacy_method='chi_sq')
    mu, V = pca_fit(Xn, NQ * DEPTH)
    Xn_p = (Xn - mu) @ V
    Xa_pool = (np.nan_to_num(np.vstack([d['windows'] for d in atk.values()])) - mu) @ V
    qc = QuantumCircuit(NQ, DEPTH)
    theta = design_theta(Xn_p, Xa_pool, qc, SEED)

    d = {'W_L': 10, 'nom_mean': nom_mean, 'nom_std': nom_std}
    Xf_p = (fresh_noms('CNC', SEED, d, n=30) - mu) @ V          # fresh nominal
    Xa_sin = (np.nan_to_num(atk['command']['windows']) - mu) @ V  # sinusoidal attack

    rng = np.random.RandomState(2026)
    rr = [tuple(sorted(rng.choice(len(Xn_p), 2, replace=False))) for _ in range(N_PAIRS)]
    rn = [(int(rng.randint(len(Xn_p))), int(rng.randint(len(Xf_p)))) for _ in range(N_PAIRS)]
    ra = [(int(rng.randint(len(Xn_p))), int(rng.randint(len(Xa_sin)))) for _ in range(N_PAIRS)]

    sets = {'ref': Xn_p, 'nom': Xf_p, 'atk': Xa_sin}
    bands = {'ref-ref': [('ref', int(i), 'ref', int(j)) for i, j in rr],
             'ref-nom': [('ref', i, 'nom', j) for i, j in rn],
             'ref-atk': [('ref', i, 'atk', j) for i, j in ra]}

    ref_fid = {}
    for c in (1.0, 0.1):
        ref_fid[str(c)] = {}
        for bname, plist in bands.items():
            vals = [qc.fidelity_kernel(c * sets[sa][i], c * sets[sb][j], theta)
                    for sa, i, sb, j in plist]
            ref_fid[str(c)][bname] = [float(v) for v in vals]
            print(f"c={c:<4} {bname}: exact k = {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    json.dump({
        'benchmark': 'CNC', 'seed': SEED, 'n_qubits': NQ, 'depth': DEPTH,
        'theta': theta.tolist(),
        'vectors': {k: v.tolist() for k, v in sets.items()},
        'bands': bands, 'scales': [1.0, 0.1],
        'ref_fidelity': ref_fid,
        'note': 'canonical sign-deterministic pipeline; encoding: per layer l, '
                'per qubit i: RY(theta[l,i,0]*x[(l*4+i)%12]) then RZ(theta[l,i,1]*x[...]); '
                'entangler CNOT(0,1),(1,2),(2,3),(3,0) after each encoding layer',
    }, open(OUT, 'w'))
    print(f"\nsaved {OUT}  ({3 * N_PAIRS} pairs x 2 scales = {6 * N_PAIRS} circuits)")


if __name__ == '__main__':
    main()
