"""
stage1_gen_full_lc_v2.py — stage 1, parallelised ACROSS SOURCES.

Why v2: v1 processed one source at a time with rank 0 doing every DRW fit
serially (23 ranks idle) and collectives per source — so one pathological
source (crash or slow fit) hung the entire job at a bcast, and 300 sources
meant ~13h of serial fitting. v2:

  * round-robin sources across ranks; each rank does load -> drw_fit ->
    gpSimFull -> write end-to-end for its own sources. NO collectives in the
    work loop: a bad source costs one log line, never a hang.
  * per-source try/except: any exception is logged, source skipped (no
    sentinel), run continues.
  * MIN_EPOCHS guard: LCs with too few epochs after masking are skipped
    up-front (v1 crashed on a 1-epoch LC via drw_fit's np.min on empty diff).
  * DRW_TIMEOUT: SIGALRM around drw_fit; a fit exceeding it is logged and
    the source skipped.

Outputs, filenames, sentinels, and per-sim seeding are IDENTICAL to v1
(seed = md5(ra_dec_band) + global_sim_index), so stages 2/3 and all checkers
work unchanged. Each source now yields ONE parquet
(sim_{ra}_{dec}_z{band}_rank{owner}_full.parquet) instead of 24 — the
downstream glob rank*_full.parquet matches either layout.

Usage:
    mpirun -n <nranks> python stage1_gen_full_lc_v2.py
"""

import os
import re
import glob
import time
import signal
import hashlib
import numpy as np
import pandas as pd
from mpi4py import MPI

from eztao.ts import drw_fit
from eztao.carma import DRW_term
from eztao.ts import gpSimFull

from VarTools import _find_target_obj

# ── configuration ─────────────────────────────────────────────────────────────
N_SIMS      = 120
N_SOURCES   = 10            # shuffled-prefix subsample; raise to extend
SHUFFLE_SEED = 42            # fixed: same subset every submission
MIN_EPOCHS  = 30             # skip sources with fewer masked epochs
DRW_TIMEOUT = 300            # seconds per drw_fit before giving up
DATA_PATH   = os.environ["HOME"] + "/BAT_results/"
# OUTPUT_PATH = os.environ["HOME"] + "/results/full_lc/"
OUTPUT_PATH = os.environ["HOME"] + "/results/20yr_lc/full_lc/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ── helpers (identical semantics to v1) ───────────────────────────────────────

def radec_filename(filename):
    match  = re.search(r"(\d+\.\d+)_([-+]?\d+\.\d+)", filename)
    match2 = re.search(r"_z(\w)_", filename)
    band   = match2.group(1)
    if match:
        return float(match.group(1)), float(match.group(2)), band
    raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")

from pathlib import Path
from astropy.coordinates import SkyCoord

def load_original_lc(ra, dec, path):
    ztf = pd.read_parquet(path)
    coord = SkyCoord(ra, dec, unit='deg')
    tgt = _find_target_obj(Path(path), coord)
    if tgt is None:
        return None, "no target found"
    ztf1 = ztf[ztf["object_index"] == tgt]
    mask = ztf1["MERR_4_TOT_AB"] < 0.5
    if "qid" in ztf1.columns:
        qid   = ztf1["qid"].mode().iloc[0]
        ccdid = ztf1["ccdid"].mode().iloc[0]
        field = ztf1["field"].mode().iloc[0]
        mask &= ((ztf1["qid"] == qid) & (ztf1["ccdid"] == ccdid)
                 & (ztf1["field"] == field))
    ztf1 = ztf1[mask]
    if len(ztf1) < MIN_EPOCHS:
        return None, f"only {len(ztf1)} epochs after masking (<{MIN_EPOCHS})"
    t = ztf1.OBSMJD.values.copy()
    t -= t[0]
    return (t, ztf1["MAG_4_TOT_AB"].values, ztf1["MERR_4_TOT_AB"].values), ""


def source_seed(ra, dec, band):
    s = f"{ra}_{dec}_{band}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def sentinel_path(ra, dec, band):
    return OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}.stage1.done"


class DRWTimeout(Exception):
    pass

def _alarm(signum, frame):
    raise DRWTimeout()


def drw_fit_timed(t, mag, err, seconds):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return drw_fit(t, mag, err)
    finally:
        signal.alarm(0)


# ── per-source, end-to-end on the owning rank ─────────────────────────────────

def process_source(f):
    ra, dec, band = radec_filename(f)
    tag = f"{ra}_{dec}_z{band}"
    if os.path.exists(sentinel_path(ra, dec, band)):
        return "skip"

    loaded, why = load_original_lc(ra, dec, f)
    if loaded is None:
        print(f"[rank {rank}] {tag}: SKIP ({why})", flush=True)
        return "nodata"
    time_arr, mag, mag_err = loaded

    t0 = time.time()
    try:
        best_fit = drw_fit_timed(time_arr, mag, mag_err, DRW_TIMEOUT)
    except DRWTimeout:
        print(f"[rank {rank}] {tag}: SKIP (drw_fit > {DRW_TIMEOUT}s)", flush=True)
        return "timeout"
    print(f"[rank {rank}] {tag}: drw_fit {time.time()-t0:.1f}s {best_fit}", flush=True)

    # dur, N   = float(time_arr[-1]), len(time_arr)
    dur = 20 * 365.25   # 7305 days
    N = int(dur) + 1    # 7306 → exactly 1-day spacing
    mag_med  = float(np.median(mag))
    SNR      = best_fit[0] / float(np.median(mag_err))
    kernel   = DRW_term(*np.log(best_fit))
    seed0    = source_seed(ra, dec, band)

    t1 = time.time()
    t_even = None
    mags = np.empty((N_SIMS, N))
    errs = np.empty((N_SIMS, N))
    for gidx in range(N_SIMS):
        t_b, y_b, e_b = gpSimFull(kernel, SNR, dur, N, nLC=1, log_flux=True,
                                  lc_seed=(seed0 + gidx) & 0x7FFFFFFF)
        if t_even is None:
            t_even = np.ravel(t_b)
        mags[gidx] = np.ravel(y_b) + mag_med
        errs[gidx] = np.ravel(e_b)

    df = pd.DataFrame({
        "object_index": np.repeat(np.arange(N_SIMS), N),
        "time": np.tile(t_even, N_SIMS),
        "mag":  mags.ravel(),
        "err":  errs.ravel(),
    })
    path = OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}_rank{rank}_full.parquet"
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    open(sentinel_path(ra, dec, band), "w").close()
    print(f"[rank {rank}] {tag}: {N_SIMS} sims x {N} epochs "
          f"(gpSim {time.time()-t1:.1f}s) -> done", flush=True)
    return "done"


def main():
    job_start = time.time()
    import random

    # all_files = sorted(glob.glob(DATA_PATH + "*z[gri]_merged.parquet"))
    all_files = sorted(glob.glob(DATA_PATH + "0.61008_+3.35193_z[gri]_merged.parquet"))
    random.Random(SHUFFLE_SEED).shuffle(all_files)
    subset = all_files#[:N_SOURCES]

    mine = [f for i, f in enumerate(subset) if i % nrank == rank]
    if rank == 0:
        print(f"[rank 0] {len(all_files)} total, subset {len(subset)}, "
              f"{nrank} ranks (~{len(subset)//max(nrank,1)}/rank)", flush=True)

    counts = {"done": 0, "skip": 0, "nodata": 0, "timeout": 0, "error": 0}
    for f in mine:
        try:
            counts[process_source(f)] += 1
        except Exception as e:
            print(f"[rank {rank}] ERROR {os.path.basename(f)}: "
                  f"{type(e).__name__}: {e}", flush=True)
            counts["error"] += 1

    print(f"[rank {rank}] finished: {counts}", flush=True)
    comm.Barrier()
    if rank == 0:
        n_done = len(glob.glob(OUTPUT_PATH + "*.stage1.done"))
        print(f"[rank 0] stage 1 v2 walltime {time.time()-job_start:.1f}s; "
              f"{n_done} total sentinels", flush=True)


if __name__ == "__main__":
    main()