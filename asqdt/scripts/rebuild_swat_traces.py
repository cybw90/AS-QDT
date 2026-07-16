#!/usr/bin/env python3
"""
Rebuild the original chronological SWaT traces from label-split files.

Some SWaT redistributions split the original recordings by label (all
normal rows in one file, all attack rows in another), which destroys the
temporal interleaving of the ~36 attack scenarios and makes per-scenario
evaluation impossible. This script reconstructs the original chronology
from the row timestamps.

Usage:
  python -m asqdt.scripts.rebuild_swat_traces dataset/merged.csv \
      --out-normal dataset/normal_rebuilt.csv \
      --out-attack dataset/attack_rebuilt.csv

Steps: parse Timestamp (day-first), sort chronologically, drop duplicate
timestamps (keep first), split at `margin` hours before the first
attack-labeled row: everything before -> normal trace (labels dropped),
everything after -> attack trace (labels kept). Prints the recovered row,
attack-second, and contiguous-scenario counts (~30-40 scenarios expected
for SWaT.A1&A2), computes the SHA-256 of each reconstructed trace, and
writes a manifest JSON recording those counts and checksums so a reader
with the licensed raw files can confirm a bit-identical reconstruction.
"""
import sys, os, argparse, hashlib, json
import numpy as np
import pandas as pd


def _sha256(path, buf=1 << 20):
    """Stream a file through SHA-256 so large CSVs don't load into memory."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(buf), b''):
            h.update(chunk)
    return h.hexdigest()

ATTACK_TOKENS = ('attack', 'a ttack', '1', 'anomaly', 'true')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('merged')
    ap.add_argument('--out-normal', default='normal_rebuilt.csv')
    ap.add_argument('--out-attack', default='attack_rebuilt.csv')
    ap.add_argument('--margin-hours', type=float, default=2.0,
                    help='normal margin kept in front of the attack trace')
    ap.add_argument('--manifest', default='swat_reconstruction_manifest.json',
                    help='where to write recovered counts and SHA-256 checksums')
    a = ap.parse_args()

    print(f"reading {a.merged} ...")
    df = pd.read_csv(a.merged)
    df.columns = [str(c).strip() for c in df.columns]
    ts_col = next(c for c in df.columns if c.lower() == 'timestamp')
    label_col = next(c for c in df.columns
                     if c.lower() in ('normal/attack', 'normal_attack', 'label', 'attack'))

    df[ts_col] = pd.to_datetime(df[ts_col].astype(str).str.strip(),
                                dayfirst=True, format='mixed')
    n0 = len(df)
    df = df.sort_values(ts_col, kind='mergesort')
    df = df.drop_duplicates(subset=[ts_col], keep='first').reset_index(drop=True)
    print(f"sorted {n0} rows -> {len(df)} after de-duplicating timestamps")

    lab = df[label_col].astype(str).str.strip().str.lower().isin(ATTACK_TOKENS).to_numpy()
    if lab.sum() == 0:
        sys.exit("no attack-labeled rows found -- check the label column")
    first_atk = df[ts_col].iloc[int(np.argmax(lab))]
    split_t = first_atk - pd.Timedelta(hours=a.margin_hours)
    print(f"first attack at {first_atk}; splitting at {split_t}")

    normal = df[df[ts_col] < split_t]
    attack = df[df[ts_col] >= split_t]

    # report recovered scenarios + gaps (sanity)
    la = attack[label_col].astype(str).str.strip().str.lower().isin(ATTACK_TOKENS).to_numpy()
    segs = int((la[1:] & ~la[:-1]).sum() + la[0])
    gaps = attack[ts_col].diff().dt.total_seconds().fillna(1)
    print(f"normal trace: {len(normal)} rows "
          f"({normal[ts_col].iloc[0]} .. {normal[ts_col].iloc[-1]})")
    print(f"attack trace: {len(attack)} rows, {la.sum()} attack samples, "
          f"{segs} contiguous attack scenarios "
          f"({attack[ts_col].iloc[0]} .. {attack[ts_col].iloc[-1]})")
    if (gaps > 60).any():
        print(f"  note: {(gaps > 60).sum()} recording gap(s) > 60 s inside the "
              "attack trace (expected between collection days)")
    if segs < 5:
        print("  WARNING: fewer than 5 scenarios recovered -- the merged file "
              "may not contain the normal rows of the attack days.")

    normal.drop(columns=[label_col]).to_csv(a.out_normal, index=False)
    attack.to_csv(a.out_attack, index=False)
    print(f"wrote {a.out_normal} and {a.out_attack}")

    # SHA-256 of each reconstructed trace, so a reader with the licensed raw
    # files can confirm a bit-identical reconstruction (Supplementary Q).
    normal_sha = _sha256(a.out_normal)
    attack_sha = _sha256(a.out_attack)
    print(f"sha256({a.out_normal}) = {normal_sha}")
    print(f"sha256({a.out_attack}) = {attack_sha}")

    manifest = {
        'normal_rows': int(len(normal)),
        'attack_rows': int(len(attack)),
        'attack_seconds': int(la.sum()),
        'n_contiguous_scenarios': int(segs),
        'out_normal': a.out_normal,
        'out_attack': a.out_attack,
        'sha256_normal': normal_sha,
        'sha256_attack': attack_sha,
    }
    manifest_path = a.manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote reconstruction manifest {manifest_path}")


if __name__ == '__main__':
    main()
