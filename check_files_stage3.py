"""
check_stage3_format.py — validate the stage-3 cache FORMATS and the
parse_to_dict / format_SFdf round-trip, before trusting plots or refits.

Usage (serial, compute node, conda env):
    python check_stage3_format.py                    # all sources with fit files
    python check_stage3_format.py <ra> <dec> <band>  # one source, verbose
    python check_stage3_format.py --selftest         # synthetic end-to-end test
                                                     # (no cluster data needed)

Checks
------
JSONL fit cache:
  - every line is valid JSON (incl. NaN handling on invalid-fit rows)
  - exact key set: object_index, cadence, valid, fail_reason + 7 FIT_KEYS
  - types: int / str('full'|'crop') / bool / str / floats
  - no duplicate (object_index, cadence)
SF parquet cache:
  - columns object_index, cadence, SF, SFmaxerr, SFminerr; strings non-empty
  - no '...' elision marker in any stored string
parse_to_dict round-trip (per sampled row):
  - reconstructs a Series with the expected bin count
  - CategoricalIndex of right-closed Intervals (what SF_linmix consumes)
SF_linmix preprocessing dry-run (NO MCMC) — replicates SF_linmix exactly:
  - pd.IntervalIndex(SF.index) works
  - [1,365] window mask + !=0 + dropna leaves > 3 points
  - SFmaxerr/SFminerr .reindex(sf_mag.index) yields NO NaN
    (catches float-mismatch between separately parsed interval indexes)
format_SFdf:
  - row -> dict with three parsed Series under the right keys
"""

import os
import re
import sys
import glob
import json
import numpy as np
import pandas as pd

from newSF import parse_to_dict, format_SFdf

CACHE    = os.environ["HOME"] + "/results/sf_cache/"
CADENCES = ("full", "crop")
N_SIMS   = 120
mode = 'spl'
FIT_KEYS = [
   f"A_1_{mode}", f"A_365_{mode}", f"A_maxerr_{mode}", f"A_minerr_{mode}",
    f"gamma_{mode}", f"gamma_maxerr_{mode}", f"gamma_minerr_{mode}",
]
EXPECTED_JSON_KEYS = {"object_index", "cadence", "valid", "fail_reason", *FIT_KEYS}
N_SAMPLE_ROWS = 8   # SF rows per source to deep-check


def check_jsonl(ra, dec, band, verbose=False):
    ok, rows = True, []
    files = sorted(glob.glob(CACHE + f"fit_{ra}_{dec}_z{band}_rank*.jsonl"))
    if not files:
        print(f"  [FAIL] no fit JSONL files")
        return False, None
    for p in files:
        for i, line in enumerate(open(p)):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [FAIL] {os.path.basename(p)}:{i+1} bad JSON: {e}")
                ok = False
                continue
            keys = set(r.keys())
            if keys != EXPECTED_JSON_KEYS:
                print(f"  [FAIL] {os.path.basename(p)}:{i+1} keys "
                      f"missing={EXPECTED_JSON_KEYS-keys} extra={keys-EXPECTED_JSON_KEYS}")
                ok = False
            if not isinstance(r.get("object_index"), int):
                print(f"  [FAIL] {os.path.basename(p)}:{i+1} object_index not int")
                ok = False
            if r.get("cadence") not in CADENCES:
                print(f"  [FAIL] {os.path.basename(p)}:{i+1} cadence={r.get('cadence')!r}")
                ok = False
            if not isinstance(r.get("valid"), bool):
                print(f"  [FAIL] {os.path.basename(p)}:{i+1} valid not bool")
                ok = False
            if r.get("valid"):
                bad = [k for k in FIT_KEYS
                       if not isinstance(r.get(k), (int, float)) or not np.isfinite(r[k])]
                if bad:
                    print(f"  [FAIL] {os.path.basename(p)}:{i+1} valid row has "
                          f"non-finite {bad}")
                    ok = False
            rows.append(r)

    pairs = [(r["object_index"], r["cadence"]) for r in rows]
    if len(pairs) != len(set(pairs)):
        print(f"  [FAIL] {len(pairs)-len(set(pairs))} duplicate (oi, cadence) rows")
        ok = False
    if verbose:
        nv = sum(r["valid"] for r in rows)
        print(f"  jsonl: {len(rows)} rows, {nv} valid, "
              f"{len(files)} rank files — {'OK' if ok else 'PROBLEMS'}")
    return ok, pd.DataFrame(rows)


def sf_linmix_dry_run(d):
    """Replicate SF_linmix preprocessing exactly, up to (not including) MCMC.
    Returns (would_fit: bool, reason: str)."""
    try:
        interval_index = pd.IntervalIndex(d["SF"].index)
    except Exception as e:
        return False, f"pd.IntervalIndex(SF.index) failed: {e}"
    window = ((interval_index.left >= 1) & (interval_index.right <= 365)
              & (d["SF"] != 0))
    sf_mag = d["SF"][window].dropna()
    if len(sf_mag) <= 3:
        return False, f"too few points in window ({len(sf_mag)}) — matches a real skip"
    maxe = d["SFmaxerr"].reindex(sf_mag.index)
    mine = d["SFminerr"].reindex(sf_mag.index)
    if maxe.isna().any() or mine.isna().any():
        return False, ("reindex produced NaN errors — interval float mismatch "
                       "between parsed SF and SFmaxerr/SFminerr")
    sf_err = (maxe + mine) / 2
    if (sf_err <= 0).any():
        return False, "non-positive combined errors (SF_linmix would skip)"
    return True, ""


def check_sf_parquet(ra, dec, band, fits_df, verbose=False):
    ok = True
    files = sorted(glob.glob(CACHE + f"sf_{ra}_{dec}_z{band}_rank*.parquet"))
    if not files:
        print(f"  [FAIL] no SF parquet files")
        return False
    sf = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    print('sf columns',sf.columns)

    need = {"object_index", "cadence", "SF", "SFmaxerr", "SFminerr"}
    if set(sf.columns) != need:
        print(f"  [FAIL] SF parquet columns {sorted(sf.columns)} != {sorted(need)}")
        return False

    for col in ("SF", "SFmaxerr", "SFminerr"):
        notstr = (~sf[col].apply(lambda x: isinstance(x, str))).sum()
        if notstr:
            print(f"  [FAIL] {notstr} non-string entries in {col}")
            ok = False
        elided = sf[col].str.contains(r"\.\.\.", regex=True).sum()
        if elided:
            print(f"  [FAIL] {elided} elided ('...') strings in {col} — "
                  f"row-truncated Series were stored!")
            ok = False

    # deep round-trip on a sample: first rows + one known-valid row
    idx = list(sf.index[:N_SAMPLE_ROWS])
    if fits_df is not None and len(fits_df):
        v = fits_df[fits_df["valid"]]
        if len(v):
            m = sf[(sf.object_index == v.iloc[0]["object_index"])
                   & (sf.cadence == v.iloc[0]["cadence"])]
            idx += list(m.index[:1])
    nbins_seen = set()
    for i in sorted(set(idx)):
        row = sf.loc[i]
        tagr = f"(oi={row['object_index']}, {row['cadence']})"
        d = {}
        parse_failed = False
        for col in ("SF", "SFmaxerr", "SFminerr"):
            parsed = parse_to_dict(row[col])["SF"]
            if not isinstance(parsed, pd.Series):
                print(f"  [FAIL] {tagr} parse_to_dict({col}) -> {type(parsed)}")
                ok = False
                parse_failed = True
                break
            if not isinstance(parsed.index, pd.CategoricalIndex):
                print(f"  [FAIL] {tagr} {col} index is {type(parsed.index)}, "
                      f"not CategoricalIndex")
                ok = False
            d[col] = parsed
        if not parse_failed:
            nbins_seen.add(len(d["SF"]))
            would, why = sf_linmix_dry_run(d)
            if fits_df is not None:
                frow = fits_df[(fits_df.object_index == row["object_index"])
                               & (fits_df.cadence == row["cadence"])]
                if len(frow):
                    was_valid = bool(frow.iloc[0]["valid"])
                    if (not would) and was_valid:
                        # cached fit succeeded but reconstruction wouldn't:
                        # round-trip regression, the serious case
                        print(f"  [FAIL] {tagr} cached fit valid but "
                              f"reconstructed SF wouldn't fit: {why}")
                        ok = False
                    elif verbose:
                        print(f"    {tagr}: round-trip ok, dry-run "
                              f"{'fit' if would else 'skip'} "
                              f"(cache: {'valid' if was_valid else 'invalid'})")
        fd = format_SFdf(row, columns=["cadence","SF", "SFmaxerr", "SFminerr"])
        # fd = format_SFdf(row)
        print('df to old_dict',fd)
        if not all(isinstance(fd.get(k), pd.Series)
                   for k in ("SF", "SFmaxerr", "SFminerr")):
            print(f"  [FAIL] {tagr} format_SFdf did not return three Series")
            ok = False

    if verbose:
        print(f"  sf parquet: {len(sf)} rows, bin counts seen {sorted(nbins_seen)} "
              f"— {'OK' if ok else 'PROBLEMS'}")
    return ok


def check_source(ra, dec, band, verbose=False):
    tag = f"{ra}_{dec}_z{band}"
    print(f"{tag}:")
    ok1, fits = check_jsonl(ra, dec, band, verbose=verbose)
    print(fits.columns)
    print(fits['_A_365_spl'])
    print(fits['_gamma_spl'].loc[0])
    print('jsonl fits', fits)
    ok2 = check_sf_parquet(ra, dec, band, fits, verbose=verbose)
    print(f"  => {'[ OK ]' if ok1 and ok2 else '[FAIL]'}")
    return ok1 and ok2


def find_sources():
    out = set()
    for p in glob.glob(CACHE + "fit_*_rank*.jsonl"):
        m = re.search(r"fit_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)_rank", os.path.basename(p))
        if m:
            out.add((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)


def selftest():
    """Synthetic end-to-end: build an SF like compute_target_stats does,
    stringify like stage 3 does, run every check above on it."""
    print("selftest: building synthetic SF cache row...")
    edges = np.logspace(np.log10(0.5), np.log10(2500), 38)
    rng = np.random.RandomState(42)
    dt = rng.uniform(0.5, 2500, 5000)
    vals = 0.2 * (dt / 365) ** 0.5 + rng.normal(0, 0.01, dt.size)
    binned = pd.Series(vals).groupby(pd.cut(pd.Series(dt), bins=edges),
                                     observed=True).mean()
    sf, maxe, mine = binned, binned * 0.1, binned * 0.08

    row = pd.Series({
        "object_index": 0, "cadence": "full",
        "SF": sf.to_string(), "SFmaxerr": maxe.to_string(),
        "SFminerr": mine.to_string(),
    })
    d = {k: parse_to_dict(row[k])["SF"] for k in ("SF", "SFmaxerr", "SFminerr")}
    assert isinstance(d["SF"].index, pd.CategoricalIndex), "index type"
    assert len(d["SF"]) == len(sf.dropna()), \
        f"bin count {len(d['SF'])} != {len(sf.dropna())}"
    orig = sf.dropna().values
    assert np.allclose(d["SF"].values, orig, rtol=1e-4), "value drift > print precision"
    would, why = sf_linmix_dry_run(d)
    assert would, f"dry run failed: {why}"
    fd = format_SFdf(row, columns=["SF", "SFmaxerr", "SFminerr"])
    assert all(isinstance(fd[k], pd.Series) for k in fd), "format_SFdf"
    j = json.dumps({"object_index": 0, "cadence": "full", "valid": False,
                    "fail_reason": "x", **{k: float("nan") for k in FIT_KEYS}})
    r = json.loads(j)
    assert np.isnan(r["_gamma_spl"]), "NaN JSON round-trip"
    print("selftest: ALL PASSED")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--selftest":
        selftest()
    elif len(a) == 3:
        check_source(float(a[0]), float(a[1]), a[2], verbose=True)
    else:
        srcs = find_sources()
        print(f"found {len(srcs)} sources with fit caches\n")
        n = sum(check_source(*s) for s in srcs)
        print(f"\n{n}/{len(srcs)} sources passed format checks")