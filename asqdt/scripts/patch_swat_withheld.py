#!/usr/bin/env python3
"""
patch_swat_withheld.py — adds withheld-half (last 18 of 35 scenarios) AUC/F1/TPR
to run_protocol's per-method results, so run_swat_validation.py reports the
independent-scenario metrics directly. Idempotent; run once from the repo root:

    python3 asqdt/scripts/patch_swat_withheld.py
    python3 asqdt/scripts/run_swat_validation.py NORMAL.csv ATTACK.csv

Each per-method line will then include auc_wh / f1_wh / tpr_wh in the saved
JSON (results dict), computed on the scenarios withheld from all design.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, '..', 'external', 'icsdata.py')

ANCHOR = "        per_scenario[name] = [(seg_ids[i], bool(s_pos[i] > tau)) for i in range(len(Xk))]"
ADDITION = """        per_scenario[name] = [(seg_ids[i], bool(s_pos[i] > tau)) for i in range(len(Xk))]
        # withheld-half metrics: scenarios beyond the design slice Xk[:max(2, len(Xk)//2)]
        nh = max(2, len(Xk) // 2)
        y_wh = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xk) - nh)])
        sc_wh = np.concatenate([s_neg, s_pos[nh:]])
        results[name].update(
            auc_wh=float(roc_auc_score(y_wh, sc_wh)),
            f1_wh=float(f1_score(y_wh, (sc_wh > tau).astype(int), zero_division=0)),
            tpr_wh=float(np.mean(s_pos[nh:] > tau)))"""


def main():
    src = open(TARGET).read()
    if "auc_wh" in src:
        print("already patched — nothing to do")
        return
    assert ANCHOR in src, "anchor line not found; icsdata.py differs from the shipped version"
    open(TARGET, 'w').write(src.replace(ANCHOR, ADDITION, 1))
    print(f"patched {os.path.normpath(TARGET)}: withheld-half auc_wh/f1_wh/tpr_wh added")


if __name__ == '__main__':
    main()
