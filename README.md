# Paper Title: Minimax Attack-Separation Margins for Digital-Twin-Based Stealthy Threat Detection: A Quantum-Kernel Feasibility Study

**A Quantum-Kernel Digital Twin for Stealthy Threat Detection**

<p align="left">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20artifact-blue">
  <img alt="Venue" src="https://img.shields.io/badge/IEEE%20TSMC-00629B">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB">
  <img alt="Quantum" src="https://img.shields.io/badge/quantum-IBM%20Quantum-6929C4">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

**Reproducibility artifact for:**
> *Minimax Attack-Separation Margins for Digital Twin-Based Stealthy Threat Detection: A Quantum-Kernel Feasibility Study*
> 
> *IEEE Transactions on Systems, Man, and Cybernetics: Systems* --- October 2026.

This repository reproduces **every number, table, and figure** reported in the paper and its supplement.

---

## Contents

- [1. Environment](#1-environment)
  - [1.1 Dependencies](#11-dependencies)
  - [1.2 Determinism](#12-determinism)
- [2. Repository Layout](#2-repository-layout)
- [3. Reproducing the Paper](#3-reproducing-the-paper)
- [4. Output Map](#4-output-map)
- [5. External Data: SWaT](#5-external-data-swat)
  - [5.1 Reconstruction and Integrity Verification](#51-reconstruction-and-integrity-verification)
- [6. Quantum Hardware](#6-quantum-hardware)
- [7. Repository Notes](#7-repository-notes)
- [8. License](#8-license)

---

## 1. Environment

The pipeline targets **Python 3.12**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core
pip install -r requirements-swat.txt       # SWaT reconstruction
pip install -r requirements-hardware.txt   # IBM Quantum re-execution
```

### 1.1 Dependencies

The core pipeline depends only on `numpy`, `scipy`, `scikit-learn`, and
`matplotlib` (pinned in `requirements.txt`); these four are sufficient to
reproduce every paper table and figure. The two other requirement files
cover functionality that is not needed for the core results: `requirements-swat.txt`
(`pandas`, for SWaT trace reconstruction) and `requirements-hardware.txt`
(`qiskit`, `qiskit-ibm-runtime`, for re-execution on IBM Quantum).

### 1.2 Determinism

The quantum feature map is sensitive to the SVD sign convention used in PCA. The
pipeline pins a deterministic `svd_flip` convention, which must be verified before
any re-run:

```bash
python -m asqdt.scripts.verify_sign_determinism   # must print PASS
```

This check PASSES on macOS/Accelerate and Linux/OpenBLAS; all quantum values
reproduce bit-identically.

---

## 2. Repository Layout

```
AS-QDT/
├── asqdt/                       # Core Python package
│   ├── config.py                # Hyperparameters, attack sets, seeds
│   ├── cpms/                    # Cyber-physical plant models + legacy detectors
│   │   ├── plant.py             #   CNC plant, observer, chi-squared, attacks
│   │   ├── cstr.py              #   CSTR plant, observer, CUSUM
│   │   ├── tank.py              #   Coupled-tank plant, observer
│   │   └── data.py              #   Dataset reproduction (one window per episode)
│   ├── detection/mmd.py         # MMD U-statistic, thresholds, detector
│   ├── quantum/                 # Quantum kernel and surrogates
│   │   ├── kernel.py            #   Statevector quantum-kernel simulator
│   │   ├── surrogate.py         #   Nyström surrogate
│   │   ├── learned_surrogate.py #   MLP surrogate
│   │   └── surrogate_variants.py
│   ├── codesign/                # Sensor / attack co-design
│   │   ├── sensors.py, sensors_v2.py   # Submodular greedy sensor selection
│   │   └── solver.py            #   Alternating co-design solver
│   ├── evaluation/              # Metrics and baselines
│   │   ├── metrics.py           #   F1, AUROC, TPR, FPR
│   │   ├── baselines_v2.py
│   │   └── fastq.py
│   ├── external/icsdata.py      # SWaT ingest (pandas)
│   ├── scripts/                 # Reproduction entry points (see §3)
│   └── results/                 # Committed result JSONs, figures, and logs
├── run_ibm_concentration.py     # IBM Quantum hardware diagnostic
├── probe_mmd.py                 # Encoding-scale probe (Supp Q)
├── ibm_job_meta.json            # Archived IBM Quantum job metadata
├── ibm_concentration_results.json  # Archived IBM Quantum raw counts
├── dataset/                     # External SWaT data goes here (not redistributed)
├── requirements.txt             # Core dependencies
├── requirements-swat.txt        # SWaT reconstruction
├── requirements-hardware.txt    # IBM Quantum re-execution
└── LICENSE                      # MIT
```

---

## 3. Reproducing the Paper

Run the following stages in order:

```bash
python -m asqdt.scripts.verify_sign_determinism      # environment check (must PASS)
python -m asqdt.scripts.run_v2_revision              # datasets + pooled kernel design
python -m asqdt.scripts.finalize_canonical           # sign tests, concentration, scale probe, in-set
python -m asqdt.scripts.run_new_experiments          # baselines, fixed-attack, adaptive, sensors, ablation
python -m asqdt.scripts.run_test_split               # disjoint fresh-attack split
python -m asqdt.scripts.run_test_split_allmethods
python -m asqdt.scripts.inset_perwindow              # in-set-only per-window
python -m asqdt.scripts.exhaustive_masks             # all 15 budgeted CNC masks
python -m asqdt.scripts.verify_suppD_membership      # realized constraint rates
python probe_mmd.py                                  # encoding-scale probe (Supp Q)
```

---

## 4. Output Map

| File | Paper artifact |
|---|---|
| `asqdt/results/tifs_v2.json` | Supp S (batch-MMD table); in-sample columns of Table III |
| `asqdt/results/v2_honest_tab8.json` | Corrected kernel columns of Tables III–IV; Spearman association |
| `asqdt/results/v2_test_split*.json` | Supp L (fresh attack split) |
| `asqdt/results/v2_inset_perwindow.json` | Supp F (in-set-only per-window) |
| `asqdt/results/v2_honest_C_*.json`, `v2_exhaustive_masks.json` | Supp P (sensor masks) |
| `asqdt/results/v2_honest_D_*.json` | Supp O (adaptive stress test) |
| `asqdt/results/v2_expE.json`, `v2_fill_remaining.json` | Supp M (surrogate ablation) |
| `asqdt/results/swat_validation.json`, `swat_canonical.json` | Supp Q (SWaT) |
| `probe_mmd_output.log` | Supp Q (encoding-scale probe: pooled objective + per-family MMD/AUC) |
| `ibm_job_meta.json`, `ibm_concentration_results.json` | Supp R (IBM Quantum, job `d97uitif47jc73a78es0`) |
| `asqdt/results/fig_3bench.pdf`, `fig_surrogate.pdf` | Paper Fig. 2; Supp M figure |

---

## 5. External Data: SWaT

The SWaT dataset is **not redistributed**. Obtain SWaT.A1 & A2 from
iTrust, SUTD: <https://itrust.sutd.edu.sg/itrust-labs_datasets/>. See
`dataset/README.md` for details.

The first 17 reconstructed scenarios form the design half; the **last 18 are withheld
from all design and tuning** and provide the independent evidence reported in the paper.

### 5.1 Reconstruction and Integrity Verification

Place the licensed merged CSV under `dataset/`, then run:

```bash
python -m asqdt.scripts.rebuild_swat_traces dataset/merged.csv \
    --out-normal dataset/normal_rebuilt.csv \
    --out-attack dataset/attack_rebuilt.csv \
    --manifest asqdt/results/swat_reconstruction_manifest.json
```

The script reconstructs the original chronology from the label-split redistribution
(timestamp sort + de-duplication + margin split) and emits, for each reconstructed
trace, an **SHA-256 checksum** plus a manifest JSON recording the recovered counts.
The expected values on the reference reconstruction (see
`asqdt/results/swat_reconstruction_manifest.json` and `swat_reconstruction.log`) are:

| Quantity | Value |
|---|---|
| normal rows | 491,354 |
| attack rows | 455,365 |
| attack-labeled seconds | 54,621 |
| contiguous scenarios | 35 (from 36 documented; two-plus adjacent attacks in one uninterrupted block) |
| `sha256(normal_rebuilt.csv)` | `24d821f4…35c89cf` |
| `sha256(attack_rebuilt.csv)` | `45b0f8d1…cb70d4` |

Full 64-character digests are in the committed manifest. The checksum lets a reader
with the licensed raw files confirm a bit-identical reconstruction. Note: the digest
is exact for the released `*_rebuilt.csv`; reproducing from raw requires a matching
pandas version to reproduce an identical hash (pin via `requirements-swat.txt`).

---

## 6. Quantum Hardware

The hardware check is a **similarity-band diagnostic only** — no detection experiment
is run on hardware. Supplementary Table R reproduced from the archived raw counts
without re-running on hardware. Credentials are never stored in this repository.

---

## 7. Repository Notes

- `asqdt/results/pre_signfix/` holds superseded outputs from before the PCA sign fix — retained for provenance, **not** used by the paper.
- `run_tifs_final.py` and `fig1_margin_convergence.pdf` are legacy pre-revision entry points.
- Result-JSON key `Res-norm` denotes the paper's **PCA-feature-norm (PCA-norm)** detector.

---

## 8. License

MIT — see [`LICENSE`](LICENSE).
