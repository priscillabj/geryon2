"""
stage2_crop_lc.py

Stage 2 of the split pipeline: subsample each full even-cadence simulated LC
onto the original ZTF observation cadence (nearest-neighbour), reproducing
build_simcad from run_simulations.py. NOT a time-window truncation.

Usage:
    mpirun -n <nranks> python stage2_gen_cropped_lc.py

Parallelisation
───────────────
Across SOURCES: each rank owns a disjoint subset of sources and processes
them end-to-end. No collectives, no per-source broadcast — stage 2 is pure
IO + one cKDTree query per source. Avoids the collective-overhead / hang
issues seen at high core counts in the monolith.

Input  : ~/results/full_lc/sim_{ra}_{dec}_z{band}_rank*_full.parquet  (stage 1)
         + original ZTF parquet (for observation times)
Output : ~/results/cropped_lc/sim_{ra}_{dec}_z{band}_cropped.parquet
Sentinel: ~/results/cropped_lc/sim_{ra}_{dec}_z{band}.stage2.done

The crop index mapping depends only on (t_even, time_orig), identical for all
sims of a source, so it is computed ONCE per source and applied to every sim.
Fully deterministic given stage 1 — no RNG, no seeds.

Output schema (one row per epoch per sim, original-cadence length):
    object_index : int
    time         : f8   nearest even-cadence time to each original obs epoch
    mag          : f8
    err          : f8
"""

import os
import re
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
from mpi4py import MPI

from VarTools import _find_target_obj, radec_filename

# ── configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = os.environ["HOME"] + "/BAT_results/"
FULL_PATH    = os.environ["HOME"] + "/results/20yr_lc/full_lc/"
OUTPUT_PATH  = os.environ["HOME"] + "/results/20yr_lc/ZTF_dur/"
# OUTPUT_PATH  = os.environ["HOME"] + "/results/20yr_lc/cropped_lc/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# def radec_filename(filename):
#     match  = re.search(r"(\d+\.\d+)_([-+]?\d+\.\d+)", filename)
#     match2 = re.search(r"_z(\w)_", filename)
#     band   = match2.group(1)
#     if match:
#         return float(match.group(1)), float(match.group(2)), band
#     raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")


def load_original_times(ra, dec, path):
    """Original observation times AFTER the same masking stage 1 used.

    Must byte-match stage 1's load_original_lc filtering so the crop lands on
    the same epochs the DRW fit was based on. Returns time_orig (relative,
    starts at 0) or None.
    """
    ztf = pd.read_parquet(path)
    coord = SkyCoord(ra, dec, unit="deg")
    tgt_obj_idx = _find_target_obj(Path(path), coord)
    # tgt_obj_idx = _find_target_obj(ra, dec, ztf)
    if tgt_obj_idx is None:
        return None

    ztf1 = ztf[ztf["object_index"] == tgt_obj_idx]
    mask = ztf1["MERR_4_TOT_AB"] < 0.5

    if "qid" in ztf1.columns:
        qid   = ztf1["qid"].mode().iloc[0]
        ccdid = ztf1["ccdid"].mode().iloc[0]
        field = ztf1["field"].mode().iloc[0]
        mask &= (
            (ztf1["qid"]   == qid)   &
            (ztf1["ccdid"] == ccdid) &
            (ztf1["field"] == field)
        )

    ztf1 = ztf1[mask]
    if len(ztf1) == 0:
        return None

    time_arr = ztf1.OBSMJD.values.copy()
    time_arr -= time_arr[0]
    return time_arr


def original_file_for(ra, dec, band):
    """Locate the source's original ZTF parquet. Matches the stage-1 glob."""
    # stage 1 iterated DATA_PATH + "*sci_merged.parquet"; recover the specific
    # file by RA/DEC/band from its parsed name.
    for f in glob.glob(DATA_PATH + "*z[gri]_merged.parquet"):
        try:
            r, d, b = radec_filename(f, band=True)
        except ValueError:
            continue
        if r == ra and d == dec and b == band:
            return f
    return None


def load_full_source(ra, dec, band):
    """Reassemble all stage-1 rank slices for one source into a long df."""
    pat = FULL_PATH + f"sim_{ra}_{dec}_z{band}_rank*_full.parquet"
    files = sorted(glob.glob(pat))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def sentinel_path(ra, dec, band):
    return OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}.stage2.done"

def cropped_path(ra, dec, band):
    return OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}_cropped.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Per-source cropping
# ─────────────────────────────────────────────────────────────────────────────

def crop_source(ra, dec, band):
    """Crop every sim of one source to original cadence, write one parquet."""
    full = load_full_source(ra, dec, band)
    if full is None:
        print(f"[rank {rank}] no stage-1 data for {ra}_{dec}_z{band}", flush=True)
        return False

    orig_file = original_file_for(ra, dec, band)
    if orig_file is None:
        print(f"[rank {rank}] no original file for {ra}_{dec}_z{band}", flush=True)
        return False
    time_orig = load_original_times(ra, dec, orig_file)
    if time_orig is None:
        print(f"[rank {rank}] no usable original times for {ra}_{dec}_z{band}", flush=True)
        return False

    # even-cadence grid: identical across sims — take it from the first sim
    idx0   = full["object_index"].min()
    t_even = full[full.object_index == idx0].sort_values("time")["time"].values

    # nearest-neighbour mapping, computed ONCE for the source
    tree     = cKDTree(t_even.reshape(-1, 1))
    _, cropi = tree.query(time_orig.reshape(-1, 1))   # len = n original epochs
    t_snap   = t_even[cropi]

    # apply the same index map to every sim
    out = []
    for oi, g in full.groupby("object_index"):
        g = g.sort_values("time")
        mag = g["mag"].values
        err = g["err"].values
        out.append(pd.DataFrame({
            "object_index": oi,
            "time": t_snap,
            "mag":  mag[cropi],
            "err":  err[cropi],
        }))

    df = pd.concat(out, ignore_index=True)
    path = cropped_path(ra, dec, band)
    df.to_parquet(path, index=False)
    print(f"[rank {rank}] {ra}_{dec}_z{band}: cropped "
          f"{len(t_even)}->{len(cropi)} epochs x {full['object_index'].nunique()} sims "
          f"-> {path}", flush=True)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main — static round-robin source assignment across ranks
# ─────────────────────────────────────────────────────────────────────────────

def main():
    job_start = time.time()

    # every rank independently enumerates completed stage-1 sources
    sentinels = sorted(glob.glob(FULL_PATH + "*.stage1.done"))
    sources = []
    for p in sentinels:
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage1\.done",
                      os.path.basename(p))
        if m:
            sources.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    sources.sort()

    # round-robin assignment; skip sources already cropped
    mine = [s for i, s in enumerate(sources) if i % nrank == rank]
    if rank == 0:
        print(f"[rank 0] {len(sources)} stage-1 sources, "
              f"{nrank} ranks, ~{len(sources)//max(nrank,1)} per rank", flush=True)

    done = failed = skipped = 0
    for ra, dec, band in mine:
        if os.path.exists(sentinel_path(ra, dec, band)):
            skipped += 1
            continue
        t0 = time.time()
        try:
            ok = crop_source(ra, dec, band)
        except Exception as e:
            print(f"[rank {rank}] ERROR {ra}_{dec}_z{band}: {e}", flush=True)
            ok = False
        if ok:
            open(sentinel_path(ra, dec, band), "w").close()
            done += 1
            print(f"[rank {rank}] {ra}_{dec}_z{band} done {time.time()-t0:.1f}s", flush=True)
        else:
            failed += 1

    print(f"[rank {rank}] finished: {done} cropped, {skipped} skipped, "
          f"{failed} failed", flush=True)

    comm.Barrier()
    if rank == 0:
        print(f"[rank 0] stage 2 walltime {time.time()-job_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()