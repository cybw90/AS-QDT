#!/usr/bin/env python3
"""
verify_suppD_membership.py — regenerates the Supplementary D attack-set
membership table (seed 42): per family, min impact I, max legacy response
D_leg, and the fraction of episodes satisfying I >= rho AND D_leg <= eps.

Uses only the shipped dataset generators (same episode seeds and RNG streams
as every other experiment). Runtime: ~1 min, single core.

Run from the repository root (the directory containing asqdt/):
    python3 asqdt/scripts/verify_suppD_membership.py
"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt.cpms.data import generate_dataset
from asqdt.cpms.cstr import generate_cstr_dataset
from asqdt.cpms.tank import generate_tank_dataset

SEED = 42
LABEL = {'fdi': 'FDI', 'command': 'Sinusoidal', 'ramp': 'Ramp', 'composite': 'Composite',
         'cstr_fdi': 'FDI', 'cstr_thermal': 'Thermal', 'cstr_concentration': 'Conc.',
         'tank_bias': 'Bias', 'tank_oscillation': 'Oscill.', 'tank_drift': 'Drift'}


def report(bench, atk_data, rho, eps):
    print(f"\n  {bench}  (rho={rho}, eps={eps})")
    print(f"    {'Attack':<12} {'min I':>10} {'max D':>10}  Frac")
    for name, d in atk_data.items():
        I = np.asarray(d['impacts'], float)
        D = np.asarray(d['legacy_responses'], float)
        ok = int(np.sum((I >= rho) & (D <= eps)))
        print(f"    {LABEL.get(name, name):<12} {I.min():>10.2f} {D.max():>10.1f}  {ok}/{len(I)}")


def main():
    print("=" * 60)
    print(" Supplementary D — realized attack-set membership (seed 42)")
    print("=" * 60)

    _, atk, _, _, _ = generate_dataset(mismatch_frac=0.05, seed=SEED,
                                       W_L=10, E_max=2.0, legacy_method='chi_sq')
    report("CNC", atk, rho=0.5, eps=100.0)

    nom, atk = generate_cstr_dataset(mismatch_frac=0.05, n_nom=20,
                                     n_atk=10, H=120, seed=SEED)
    report("CSTR", atk, rho=0.2, eps=300.0)

    nom, atk = generate_tank_dataset(mismatch_frac=0.05, n_nom=20,
                                     n_atk=10, H=120, seed=SEED)
    report("Tank", atk, rho=0.1, eps=50.0)

    print("\nCompare against the Supplementary D table (paper values):")
    print("  CNC : FDI 62.8/87.2 10/10 | Sin 3.4/42.7 10/10 | "
          "Ramp 40.4/125.7 4/10 | Comp 49.6/71.0 10/10")
    print("  CSTR: FDI 0.21/1528.8 7/10 | Thermal 0.19/33.8 9/10 | "
          "Conc 0.20/1023.1 6/10")
    print("  Tank: Bias 7.81/54.9 9/10 | Oscill 6.33/54.9 9/10 | "
          "Drift 8.64/54.9 9/10")


if __name__ == '__main__':
    main()
