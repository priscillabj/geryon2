"""
stage4_merge.py — consolidate stage-3 per-rank caches into one file per source.
This is the stage 3 -> 4 interface: downstream (rank/average) reads ONLY these.

Usage (serial — ppn=1 batch job or interactive):
    python stage4_merge.py            # merge fit JSONLs for all sentineled sources
    python stage4_merge.py --sf       # also merge SF string parquets
    python stage4_merge.py --force    # re-merge even if output exists

Input : ~/results/sf_cache/fit_{ra}_{dec}_z{band}_rank*.jsonl
        ~/results/sf_cache/sf_{ra}_{dec}_z{band}_rank*.parquet   (--sf)
        ~/results/sf_cache/sim_{ra}_{dec}_z{band}.stage3.done    (source gating)
Output: ~/results/stage4/fit_{ra}_{dec}_z{band}.parquet
        ~/results/stage4/sf_{ra}_{dec}_z{band}.parquet           (--sf)

A source is merged only if its stage-3 sentinel exists AND the merged fit set
is exactly N_SIMS x len(CADENCES) unique (object_index, cadence) pairs.
Incomplete or duplicated sources are reported and skipped — a partial stage 3
cannot silently become a short stage-4 input.
"""

import os
import re
import sys
import glob
import json
import numpy as np
import pandas as pd

CACHE    = os.environ["HOME"] + "/results/sf_cache/"
OUT      = os.environ["HOME"] + "/results/stage4/"
N_SIMS   = 120
CADENCES = ("full", "crop")
mode = 'spl'
FIT_KEYS = [
   f"A_1_{mode}", f"A_365_{mode}", f"A_maxerr_{mode}", f"A_minerr_{mode}",
    f"gamma_{mode}", f"gamma_maxerr_{mode}", f"gamma_minerr_{mode}",
]

MERGE_SF = "--sf" in sys.argv
FORCE    = "--force" in sys.argv

os.makedirs(OUT, exist_ok=True)


def sentineled_sources():
    out = []
    for p in glob.glob(CACHE + "*.stage3.done"):
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage3\.done",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)


def merge_fits(ra, dec, band):
    tag = f"{ra}_{dec}_z{band}"
    dst = OUT + f"fit_{tag}.parquet"
    if os.path.exists(dst) and not FORCE:
        print(f"[skip] {tag}: merged fit exists (use --force to redo)")
        return True

    rows = []
    for p in sorted(glob.glob(CACHE + f"fit_{tag}_rank*.jsonl")):
        with open(p) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[FAIL] {tag}: bad JSON {os.path.basename(p)}:{i+1}")
                    return False
    if not rows:
        print(f"[FAIL] {tag}: no fit rows")
        return False

    df = pd.DataFrame(rows)

    # completeness + uniqueness gate
    pairs = list(zip(df["object_index"], df["cadence"]))
    expected = {(oi, c) for oi in range(N_SIMS) for c in CADENCES}
    dupes = len(pairs) - len(set(pairs))
    missing = expected - set(pairs)
    extra = set(pairs) - expected
    if dupes or missing or extra:
        print(f"[FAIL] {tag}: dupes={dupes} missing={len(missing)} "
              f"unexpected={len(extra)} — not merging")
        return False

    # typed, ordered output
    df = df.astype({"object_index": np.int32, "cadence": "category",
                    "valid": bool, "fail_reason": str,
                    **{k: np.float64 for k in FIT_KEYS}})
    
    bad = ~np.isfinite(df[FIT_KEYS]).all(axis=1) & df["valid"]
    if bad.any():
        df.loc[bad, "valid"] = False
        df.loc[bad, "fail_reason"] = "non-finite fit values (sanitized at merge)"
        print(f"       {tag}: {int(bad.sum())} rows demoted to invalid (non-finite)")

    df = df.sort_values(["object_index", "cadence"]).reset_index(drop=True)

    tmp = dst + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)
    nv = int(df["valid"].sum())
    print(f"[ ok ] {tag}: {len(df)} fits ({nv} valid) -> {os.path.basename(dst)}")
    return True


def merge_sf(ra, dec, band):
    tag = f"{ra}_{dec}_z{band}"
    dst = OUT + f"sf_{tag}.parquet"
    if os.path.exists(dst) and not FORCE:
        print(f"[skip] {tag}: merged SF exists")
        return True
    files = sorted(glob.glob(CACHE + f"sf_{tag}_rank*.parquet"))
    if not files:
        print(f"[FAIL] {tag}: no SF rank parquets")
        return False
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.sort_values(["object_index", "cadence"]).reset_index(drop=True)
    tmp = dst + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)
    print(f"[ ok ] {tag}: {len(df)} SF rows -> {os.path.basename(dst)}")
    return True


if __name__ == "__main__":
    srcs = sentineled_sources()
    print(f"{len(srcs)} stage-3-complete sources; merge_sf={MERGE_SF}\n")
    ok = 0
    for s in srcs:
        good = merge_fits(*s)
        if good and MERGE_SF:
            good = merge_sf(*s)
        ok += bool(good)
    print(f"\n{ok}/{len(srcs)} sources merged")