"""
check_stage3.py — validate stage 3 fit + SF caches.

Usage (serial, on a compute node — needs newSF/linmix for --determinism):
    python check_stage3.py                     # completeness + validity, all sources
    python check_stage3.py 288.4 -12.1 g       # detail for one source
    python check_stage3.py --determinism 288.4 -12.1 g [oi] [cadence]
        # re-run ONE fit from the cached SF with its deterministic seed and
        # confirm it reproduces the cached gamma byte-for-byte.
        # Defaults: oi=0, cadence=full.

Checks per source (all rank files combined):
  - fit rows == N_SIMS x 2 cadences, each (object_index, cadence) exactly once
  - SF cache rows match fit rows (every fit has its SF snapshot)
  - valid-rate report; groups fail_reasons
  - flat-LC signature: high invalid rate with 'SF_linmix returned None'
    concentrated in BOTH cadences of the same sims points at stage 1's DRW
    amplitude, not stage 3
  - gamma sanity: valid gammas finite, and distribution summary printed
    (DRW expectation: gamma ~ 0.5 at short lags; wildly different medians
    are a science flag, not a pipeline failure)
"""

import os
import re
import sys
import glob
import json
import hashlib
import numpy as np
import pandas as pd

OUTPUT_PATH = os.environ["HOME"] + "/results/sf_cache/"
N_SIMS   = 120
CADENCES = ("full", "crop")
mode = 'spl'
FIT_KEYS = [
   f"A_1_{mode}", f"A_365_{mode}", f"A_maxerr_{mode}", f"A_minerr_{mode}",
    f"gamma_{mode}", f"gamma_maxerr_{mode}", f"gamma_minerr_{mode}",
]


def linmix_seed(ra, dec, band, oi, cadence):
    """Must match stage3_sf_linmix.py exactly."""
    s = f"{ra}_{dec}_{band}_{oi}_{cadence}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16) & 0x7FFFFFFF


def find_sources():
    out = []
    for p in glob.glob(OUTPUT_PATH + "*.stage3.done"):
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage3\.done",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)


def load_fits(ra, dec, band):
    rows = []
    for p in sorted(glob.glob(OUTPUT_PATH + f"fit_{ra}_{dec}_z{band}_rank*.jsonl")):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [warn] torn line in {os.path.basename(p)} (ignored)")
    return pd.DataFrame(rows) if rows else None


def load_sf(ra, dec, band):
    files = sorted(glob.glob(OUTPUT_PATH + f"sf_{ra}_{dec}_z{band}_rank*.parquet"))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def check_source(ra, dec, band, verbose=False):
    tag = f"{ra}_{dec}_z{band}"
    fits = load_fits(ra, dec, band)
    sf   = load_sf(ra, dec, band)
    if fits is None:
        print(f"[FAIL] {tag}: no fit JSONL files")
        return False, sf, None

    ok = True
    expected = N_SIMS * len(CADENCES)

    # completeness + uniqueness
    pairs = list(zip(fits["object_index"], fits["cadence"]))
    if len(pairs) != len(set(pairs)):
        dupes = len(pairs) - len(set(pairs))
        print(f"[FAIL] {tag}: {dupes} duplicate (object_index, cadence) fit rows")
        ok = False
    missing = {(oi, c) for oi in range(N_SIMS) for c in CADENCES} - set(pairs)
    if missing:
        print(f"[FAIL] {tag}: {len(missing)}/{expected} fits missing "
              f"(e.g. {sorted(missing)[:5]})")
        ok = False

    # SF cache alignment
    if sf is None:
        print(f"[FAIL] {tag}: no SF cache parquet files")
        ok = False
    else:
        sf_pairs = set(zip(sf["object_index"], sf["cadence"]))
        no_sf = set(pairs) - sf_pairs
        if no_sf:
            print(f"[FAIL] {tag}: {len(no_sf)} fits without cached SF")
            ok = False

    # validity report
    n_valid = int(fits["valid"].sum())
    rate = n_valid / len(fits) if len(fits) else 0
    reasons = fits.loc[~fits["valid"], "fail_reason"].value_counts()

    # flat-LC signature: invalid in BOTH cadences of the same sim
    inv = fits[~fits["valid"]]
    both_bad = (inv.groupby("object_index")["cadence"].nunique() == 2).sum()

    # gamma sanity on valid fits
    v = fits[fits["valid"]]
    gam_finite = np.isfinite(v["_gamma_spl"]).all() if len(v) else True
    if not gam_finite:
        print(f"[FAIL] {tag}: non-finite gamma in rows marked valid")
        ok = False

    status = "[ OK ]" if ok else "[FAIL]"
    print(f"{status} {tag}: {len(fits)}/{expected} fits, "
          f"{n_valid} valid ({rate:.0%})")
    if len(reasons):
        for r, c in reasons.items():
            print(f"         invalid x{c}: {r}")
    if both_bad > 5:
        print(f"  [WARN] {tag}: {both_bad} sims invalid in BOTH cadences — "
              f"flat-LC / DRW-amplitude signature; check stage 1 std(mag), "
              f"not stage 3")
    if verbose and len(v):
        for cad in CADENCES:
            g = v.loc[v["cadence"] == cad, "_gamma_spl"]
            if len(g):
                print(f"         gamma[{cad}]: median {g.median():.3f}, "
                      f"[16,84]% = [{g.quantile(.16):.3f}, {g.quantile(.84):.3f}], "
                      f"n={len(g)}")
    return ok, sf, fits


def determinism_check(ra, dec, band, oi=0, cadence="full"):
    """Re-run one fit from cached SF with its seed; compare to cached gamma.

    Requires newSF (parse_to_dict, SF_linmix) and linmix — run in the conda
    env on a compute node.
    """
    from newSF import parse_to_dict, SF_linmix

    tag = f"{ra}_{dec}_z{band}"
    fits = load_fits(ra, dec, band)
    sf   = load_sf(ra, dec, band)
    if fits is None or sf is None:
        print(f"[FAIL] {tag}: missing fit or SF cache")
        return False

    frow = fits[(fits.object_index == oi) & (fits.cadence == cadence)]
    srow = sf[(sf.object_index == oi) & (sf.cadence == cadence)]
    if len(frow) == 0 or len(srow) == 0:
        print(f"[FAIL] {tag} sim {oi} {cadence}: not in cache")
        return False
    frow, srow = frow.iloc[0], srow.iloc[0]

    if not frow["valid"]:
        print(f"[skip] {tag} sim {oi} {cadence}: cached fit was invalid "
              f"({frow['fail_reason']}) — pick a valid one")
        return False

    sf_dict = {
        "SF":       parse_to_dict(srow["SF"])["SF"],
        "SFmaxerr": parse_to_dict(srow["SFmaxerr"])["SF"],
        "SFminerr": parse_to_dict(srow["SFminerr"])["SF"],
    }

    np.random.seed(linmix_seed(ra, dec, band, oi, cadence))
    refit = SF_linmix(sf_dict)
    if refit is None:
        print(f"[FAIL] {tag} sim {oi} {cadence}: refit returned None but "
              f"cache has a valid fit — SF reconstruction mismatch?")
        return False

    print(f"\n{tag} sim {oi} {cadence} — cached vs re-run:")
    all_match = True
    for k in FIT_KEYS:
        cached, redone = float(frow[k]), float(refit[k])
        # exact reproduction expected with identical seed + identical SF input;
        # allow float round-trip tolerance from JSON serialisation
        match = np.isclose(cached, redone, rtol=1e-9, atol=1e-12)
        all_match &= match
        flag = "" if match else "   <-- MISMATCH"
        print(f"  {k:22s} cached {cached:+.9e}  rerun {redone:+.9e}{flag}")

    if all_match:
        print("[ OK ] deterministic: re-run reproduces cached fit")
    else:
        print("[FAIL] non-deterministic or SF round-trip drift — "
              "check pandas/linmix versions match the original run")
    return all_match


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "--determinism":
        ra, dec, band = float(args[1]), float(args[2]), args[3]
        oi   = int(args[4]) if len(args) > 4 else 0
        cad  = args[5] if len(args) > 5 else "full"
        determinism_check(ra, dec, band, oi, cad)

    elif len(args) == 3:
        ok, sf, fit = check_source(float(args[0]), float(args[1]), args[2], verbose=True)
        print(sf)
        # sf.to_parquet(OUTPUT_PATH + f'merged_sf_{args[0]}_{args[1]}_{args[2]}.parquet')
        print(fit)
        # fit.to_parquet(OUTPUT_PATH + f'merged_fit_{args[0]}_{args[1]}_{args[2]}.parquet')

    else:
        srcs = find_sources()
        print(f"found {len(srcs)} stage-3-complete sources\n")
        n_ok = sum(check_source(*s)[0] for s in srcs)
        print(f"\n{n_ok}/{len(srcs)} sources passed")