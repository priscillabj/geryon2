"""
stage3_sf_linmix.py

Stage 3 - the expensive stage. For each simulated LC (full + cropped) compute
the structure function (SF_wnoise) and fit it with LinMix MCMC (SF_linmix),
caching BOTH so nothing is recomputed:

  * SF cache (parquet, per source+rank): the SF / SFmaxerr / SFminerr Series
    stored via Series.to_string() - byte-compatible with parse_to_dict /
    format_SFdf in newSF.py, and elision-proof (unlike str()/to_parquet's
    implicit coercion which truncates Series > display.max_rows).
  * Fit cache (JSONL, per source+rank): one line per (object_index, cadence)
    with the seven FIT_KEYS + validity. JSONL append = true incremental write,
    no parquet rewrite, no metadata storm.

Usage:
    mpirun -n <nranks> python stage3_sf_linmix.py          # SF + fit
    mpirun -n <nranks> python stage3_sf_linmix.py --refit  # re-fit from cached SF only

Parallelisation: sims-within-source across ranks (MCMC dominates -> cores pay
off). Each rank owns a contiguous object_index slice and writes its own cache
files (no cross-rank contention). Per-sim resume: on restart, (object_index,
cadence) pairs already in the fit cache are skipped.

Determinism: LinMix uses numpy's GLOBAL RNG and exposes no seed. We seed
np.random from md5(source, object_index, cadence) immediately before each fit,
so every fit is reproducible and the cache is regenerable. SF_wnoise itself is
deterministic.

Aliasing note: SF_linmix does old_dict.update(...) - it MUTATES its input. We
snapshot the SF strings BEFORE calling it so the cached SF is the pristine
structure function, not the fit-augmented dict.
"""

import os
import re
import sys
import glob
import json
import time
import signal
import hashlib
import numpy as np
import pandas as pd
from mpi4py import MPI

from newSF import SF_wnoise, SF_linmix

# -- configuration --
N_SIMS      = 120
FULL_PATH   = os.environ["HOME"] + "/results/20yr_lc/full_lc/"
CROP_PATH   = os.environ["HOME"] + "/results/20yr_lc/cropped_lc/"
ZTF_DUR_PATH= os.environ["HOME"] + "/results/20yr_lc/ZTF_dur/"
OUTPUT_PATH = os.environ["HOME"] + "/results/20yr_lc/sf_cache/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

FIT_KEYS = [
    "_A_1", "_A_365_spl", "_A_maxerr_spl", "_A_minerr_spl",
    "_gamma_spl", "_gamma_maxerr_spl", "_gamma_minerr_spl",
]
# CADENCES = ("full", "crop")
CADENCES = ("full", "crop", "ztf_dur")
MAX_BINS = 60   # to_string() elides above display.max_rows; guard loudly
LINMIX_TIMEOUT = 300   # seconds per SF_linmix call

REFIT = "--refit" in sys.argv

# -- MPI setup --
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# -- helpers --
class LinmixTimeout(Exception):
    pass

def _alarm(signum, frame):
    raise LinmixTimeout()

def radec_band_from_sentinel(path, stage):
    m = re.search(rf"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.{stage}\.done",
                  os.path.basename(path))
    return (float(m.group(1)), float(m.group(2)), m.group(3)) if m else None


def linmix_seed(ra, dec, band, oi, cadence):
    s = f"{ra}_{dec}_{band}_{oi}_{cadence}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16) & 0x7FFFFFFF


def slice_bounds(rank):
    base, rem = divmod(N_SIMS, nrank)
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def load_full_source(ra, dec, band):
    files = sorted(glob.glob(FULL_PATH + f"sim_{ra}_{dec}_z{band}_rank*_full.parquet"))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True) if files else None


def load_cropped_source(ra, dec, band):
    p = CROP_PATH + f"sim_{ra}_{dec}_z{band}_cropped.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else None

def load_ztfdur_source(ra, dec, band):
    p = ZTF_DUR_PATH + f"sim_{ra}_{dec}_z{band}_cropped.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else None

def fit_jsonl(ra, dec, band):
    return OUTPUT_PATH + f"fit_{ra}_{dec}_z{band}_rank{rank}.jsonl"

def sf_parquet(ra, dec, band):
    return OUTPUT_PATH + f"sf_{ra}_{dec}_z{band}_rank{rank}.parquet"

def sentinel_path(ra, dec, band):
    return OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}.stage3.done"


def load_done_fits(path):
    """(object_index, cadence) pairs already in this rank's fit JSONL."""
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((int(r["object_index"]), r["cadence"]))
                except (json.JSONDecodeError, KeyError):
                    continue   # tolerate a torn last line from a killed job
    return done

def load_done_fits_global(ra, dec, band):
    """(object_index, cadence) pairs already fit by ANY rank for this source.

    Object-index -> rank assignment (slice_bounds) depends on nranks, so a
    rerun with a different rank count reassigns some object_indices to a
    different rank than before. Checking only this rank's own JSONL would
    then miss fits another rank already did, recompute them, and append a
    second row for the same (object_index, cadence) pair — which
    stage4_merge.py's dupe check rejects outright. Union across every rank's
    file for this source so the skip-check is correct regardless of nranks.
    """
    done = set()
    for p in glob.glob(OUTPUT_PATH + f"fit_{ra}_{dec}_z{band}_rank*.jsonl"):
        done |= load_done_fits(p)
    return done

def append_fit(path, row):
    """True append - one JSON line per fit. No rewrite."""
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


# -- SF cache (parquet) - stored as to_string(), parse_to_dict-compatible --

def sf_series_to_str(s):
    if len(s) >= MAX_BINS:
        raise ValueError(
            f"SF has {len(s)} bins >= {MAX_BINS}; to_string()/str() elide beyond "
            f"display.max_rows and parse_to_dict would silently drop bins. "
            f"Widen MAX_BINS only after confirming the reader handles it.")
    return s.to_string()


def append_sf_rows(path, existing, rows):
    """SF parquet grows a few rows per sim; rewrite is cheap (<=240 rows,
    tiny strings) vs an MCMC fit. Atomic via tmp+replace."""
    existing.extend(pd.DataFrame(r, index=[0]) for r in rows)
    out = pd.concat(existing, ignore_index=True)
    tmp = path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    existing.clear()
    existing.append(out)


def load_sf_cache(path):
    if not os.path.exists(path):
        return set(), []
    df = pd.read_parquet(path)
    have = set(zip(df["object_index"].tolist(), df["cadence"].tolist()))
    return have, [df]


def reconstruct_sf_dict(row):
    """Rebuild the {'SF','SFmaxerr','SFminerr'} dict SF_linmix consumes from a
    cached SF-parquet row, using newSF.parse_to_dict for each stringified
    Series so the CategoricalIndex/closed side match exactly."""
    from newSF import parse_to_dict
    return {
        "SF":       parse_to_dict(row["SF"])["SF"],
        "SFmaxerr": parse_to_dict(row["SFmaxerr"])["SF"],
        "SFminerr": parse_to_dict(row["SFminerr"])["SF"],
    }


# -- fit --

def fit_seeded(sf_input_dict, ra, dec, band, oi, cadence):
    np.random.seed(linmix_seed(ra, dec, band, oi, cadence))
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(LINMIX_TIMEOUT)
    try:
        fit = SF_linmix(sf_input_dict)
    except LinmixTimeout:
        print(f"[rank {rank}] TIMEOUT (> {LINMIX_TIMEOUT}s) "
              f"{ra}_{dec}_z{band} sim {oi} {cadence}", flush=True)
        return False, f"SF_linmix timeout (> {LINMIX_TIMEOUT}s)", {k: np.nan for k in FIT_KEYS}
    finally:
        signal.alarm(0)
    if fit is None:
        return False, "SF_linmix returned None", {k: np.nan for k in FIT_KEYS}
    try:
        vals = {k: float(fit[k]) for k in FIT_KEYS}
    except KeyError as e:
        return False, f"missing key {e}", {k: np.nan for k in FIT_KEYS}
    if not all(np.isfinite(v) for v in vals.values()):
        return False, "non-finite fit values (amplitude overflow)", vals
    return True, "", vals


# -- per-source processing --

def process_source(ra, dec, band):
    start, stop = slice_bounds(rank)
    if stop <= start:
        return True

    fpath = fit_jsonl(ra, dec, band)
    spath = sf_parquet(ra, dec, band)
    done  = load_done_fits(fpath)
    sf_have, sf_frames = load_sf_cache(spath)

    if REFIT:
        if not os.path.exists(spath):
            print(f"[rank {rank}] {ra}_{dec}_z{band}: no SF cache to refit", flush=True)
            return False
        sf_df = pd.read_parquet(spath)
        t0, n = time.time(), 0
        for _, srow in sf_df.iterrows():
            oi, cadence = int(srow["object_index"]), srow["cadence"]
            if (oi, cadence) in done:
                continue
            sf_dict = reconstruct_sf_dict(srow)
            valid, reason, vals = fit_seeded(sf_dict, ra, dec, band, oi, cadence)
            append_fit(fpath, {"object_index": oi, "cadence": cadence,
                               "valid": valid, "fail_reason": reason, **vals})
            done.add((oi, cadence)); n += 1
        print(f"[rank {rank}] {ra}_{dec}_z{band} REFIT {n} fits {time.time()-t0:.1f}s",
              flush=True)
        return True

    full   = load_full_source(ra, dec, band)
    crop   = load_cropped_source(ra, dec, band)
    ztfdur = load_ztfdur_source(ra, dec, band)
    if full is None or crop is None or ztfdur is None:
        print(f"[rank {rank}] {ra}_{dec}_z{band}: missing full/cropped/ztf_dur input", flush=True)
        return False

    t0, n = time.time(), 0
    for oi in range(start, stop):
        f = full[full.object_index == oi].sort_values("time")
        c = crop[crop.object_index == oi].sort_values("time")
        z = ztfdur[ztfdur.object_index == oi].sort_values("time")
        if len(f) == 0 or len(c) == 0 or len(z) == 0:
            print(f"[rank {rank}] {ra}_{dec}_z{band} sim {oi}: absent in input", flush=True)
            continue

        for cadence, lc in (("full", f), ("crop", c), ("ztf_dur", z)):
            if (oi, cadence) in done:
                continue

            sf_dict = SF_wnoise(lc["mag"].values, lc["time"].values,
                                lc["err"].values, cs_all=None,
                                color="#436BAD", verbose=False)[0]

            # snapshot SF BEFORE SF_linmix mutates the dict
            if (oi, cadence) not in sf_have:
                append_sf_rows(spath, sf_frames, [{
                    "object_index": oi, "cadence": cadence,
                    "SF":       sf_series_to_str(sf_dict["SF"]),
                    "SFmaxerr": sf_series_to_str(sf_dict["SFmaxerr"]),
                    "SFminerr": sf_series_to_str(sf_dict["SFminerr"]),
                }])
                sf_have.add((oi, cadence))

            valid, reason, vals = fit_seeded(sf_dict, ra, dec, band, oi, cadence)
            append_fit(fpath, {"object_index": oi, "cadence": cadence,
                               "valid": valid, "fail_reason": reason, **vals})
            done.add((oi, cadence)); n += 1

        el = time.time() - t0
        print(f"[rank {rank}] {ra}_{dec}_z{band} sim {oi} "
              f"({n} fits, {el:.1f}s, {el/max(n,1):.1f}s/fit)", flush=True)

    print(f"[rank {rank}] {ra}_{dec}_z{band} slice [{start}:{stop}) "
          f"{n} new fits {time.time()-t0:.1f}s", flush=True)
    return True


# -- main --

def main():
    job_start = time.time()

    if rank == 0:
        s1 = {radec_band_from_sentinel(p, "stage1") for p in glob.glob(FULL_PATH + "*.stage1.done")}
        s2 = {radec_band_from_sentinel(p, "stage2") for p in glob.glob(CROP_PATH + "*.stage2.done")}
        s3 = {radec_band_from_sentinel(p, "stage2") for p in glob.glob(ZTF_DUR_PATH + "*.stage2.done")}

        ready = sorted(x for x in (s1 & s2 & s3) if x is not None)
        todo = ready if REFIT else [s for s in ready if not os.path.exists(sentinel_path(*s))]
        print(f"[rank 0] {'REFIT ' if REFIT else ''}{len(ready)} ready, "
              f"{len(todo)} to process", flush=True)
    else:
        todo = None
    todo = comm.bcast(todo, root=0)

    for ra, dec, band in todo:
        comm.Barrier()
        t0 = time.time()
        try:
            ok = process_source(ra, dec, band)
        except Exception as e:
            print(f"[rank {rank}] ERROR {ra}_{dec}_z{band}: {e}", flush=True)
            ok = False

        all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN)
        comm.Barrier()
        if rank == 0:
            if all_ok:
                open(sentinel_path(ra, dec, band), "w").close()
                print(f"[rank 0] {ra}_{dec}_z{band} SOURCE DONE {time.time()-t0:.1f}s", flush=True)
            else:
                print(f"[rank 0] {ra}_{dec}_z{band} FAILED (no sentinel)", flush=True)

    comm.Barrier()
    if rank == 0:
        print(f"[rank 0] stage 3 {'(refit) ' if REFIT else ''}walltime "
              f"{time.time()-job_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()