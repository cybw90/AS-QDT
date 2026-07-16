# AS-QDT: Minimax Attack-Separation Margins for Digital-Twin-Based Stealthy Threat Detection

Supplementary code for: *Minimax Attack-Separation Margins for Digital-Twin-Based
Stealthy Threat Detection: A Quantum-Kernel Feasibility Study*

## Reproducing the revised paper (verified pipeline)

```bash
python -m asqdt.scripts.verify_sign_determinism   # ~15 s environment check (must PASS)
python -m asqdt.scripts.run_v2_revision           # core datasets + kernel design (mp_cache.pkl)
python -m asqdt.scripts.finalize_canonical        # prints sign tests, concentration, probe, in-set (Items A-D)
python -m asqdt.scripts.run_new_experiments       # tuned baselines, fixed-attack, adaptive, sensors, ablation
python -m asqdt.scripts.run_test_split            # disjoint attack-split (Supp L)
python -m asqdt.scripts.run_test_split_allmethods
python -m asqdt.scripts.inset_perwindow           # in-set per-window (Supp F)
python -m asqdt.scripts.exhaustive_masks          # all 15 budgeted CNC masks (Supp P)
```

Key output map (revised paper):
- `results/tifs_v2.json` -> Supplementary S batch table; in-sample columns of Table III
- `results/v2_honest_tab8.json` -> corrected kernel columns of Tables III-IV; Spearman rho_S
- `results/v2_honest_D_*.json` -> Supplementary O adaptive table
- `results/v2_honest_C_*.json`, `results/v2_exhaustive_masks.json` -> Supplementary P
- `results/v2_test_split*.json` -> Supplementary L; `results/v2_inset_perwindow.json` -> Supplementary F
- `results/fig_3bench.pdf` -> Paper Fig. 2; `results/fig_surrogate.pdf` -> Supplementary M figure
- SWaT (`scripts/rebuild_swat_traces.py`, `scripts/run_swat_validation.py`) requires the external trace (Supp Q)
- IBM hardware: `export_hw_inputs.py` / `run_ibm_concentration.py` (repo root); archived raw results in
  `ibm_job_meta.json` / `ibm_concentration_results.json` (job d97uitif47jc73a78es0, ibm_fez, 4096 shots)

Total single-core runtime for the benchmark suite is about 15 min (Supplementary C.3).
The pre-revision pipeline (`run_tifs_final.py`, `fig1_margin_convergence.pdf`) is retained as legacy.
Paper Fig. 1 (architecture diagram) accompanies the manuscript sources.

## Paper–code alignment summary

| Paper claim | Code implementation |
|---|---|
| Truncated Gaussian noise, ‖w‖∞ ≤ w̄ | `np.clip()` in `plant.py`, `cstr.py`, `tank.py` |
| One window per episode at mid-episode offset | `extract_one_window()` in `data.py` |
| m=20 nominal, n=10 attack windows | `config.py`: `N_EPISODES_NOM=20`, `N_EPISODES_ATK=10` |
| Chi-squared legacy (CNC/Tank), CUSUM (CSTR) | `chi_squared_legacy()`, `cusum_legacy()` in `plant.py` |
| E_max: CNC=2, CSTR=500, Tank=20 | `config.py`: `ATTACK_SET` dict |
| ε: CNC=100, CSTR=300, Tank=50 | `config.py`: `ATTACK_SET` dict |
| Quality: CNC z=x (W_z=I₆), CSTR z=[CA,CB,T] (W_z=diag(1,1,0.01)), Tank z=[h1,h2] (W_z=I₂) | `data.py`, `cstr.py`, `tank.py` |
| CSTR observer K_o ∈ ℝ^{4×3} | `cstr.py`: `np.array([[0.3,0,0],[0,0.3,0],[0,0,0.3],[0,0,0.1]])` |
| CSTR/Tank: process noise zero | Deterministic dynamics, measurement noise only |
| Unbiased MMD U-statistic (eq. 9) | `mmd_squared()` in `mmd.py` |
| McDiarmid β = 4κ²√(½log(2/δ)(1/m+1/n)) | `compute_threshold()` in `mmd.py` |
| 4 qubits, depth 3, SPSA gradient | `QuantumCircuit(4,3)` + `spsa_gradient()` in `kernel.py` |
| Held-out bandwidth tuning | First/second half split in `run_tifs_final.py` |
| 150 perm threshold, exact paired sign test | `config.py`: `N_PERM_THRESH=150` |
| Nyström R=10, MLP (64,32,16), 500 iter | `config.py` + surrogate modules |
| Q-Direct = simulation-only oracle | Statevector kernel, not deployable online |

## Project structure

```
asqdt/
├── config.py                  # All hyperparameters
├── cpms/
│   ├── plant.py               # CNC plant, observer DT, legacy detectors, attacks
│   ├── data.py                # CNC dataset generation (one window per episode)
│   ├── cstr.py                # CSTR plant, observer, CUSUM, attacks
│   └── tank.py                # Coupled tank plant, observer, attacks
├── detection/
│   └── mmd.py                 # MMD U-statistic, threshold, detector
├── quantum/
│   ├── kernel.py              # Statevector quantum kernel simulator
│   ├── surrogate.py           # Nyström surrogate
│   └── learned_surrogate.py   # MLP surrogate
├── codesign/
│   ├── sensors.py             # Submodular greedy sensor selection
│   └── solver.py              # Alternating co-design solver
├── evaluation/
│   └── metrics.py             # F1, AUROC, TPR, FPR
├── scripts/
│   └── run_tifs_final.py      # Main experiment (produces all results)
└── results/                   # Generated outputs (3 PDFs + 1 JSON)
```

## Seeds

42, 123, 256, 789, 1024

## Requirements

Python 3.12, NumPy 2.4, SciPy 1.17, scikit-learn 1.8, matplotlib.
Single CPU core, ~3.5 min total.
