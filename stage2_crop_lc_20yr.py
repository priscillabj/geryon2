"""
stage2_crop_lc_20yr.py

Stage 2 variant for the break-reality test: crop the 20-year, 1-day-cadence
simulated LCs (stage1_gen_full_lc.py, OUTPUT_PATH=~/results/20yr_lc/full_lc/)
by REPEATING the real ZTF observation cadence back-to-back until it spans
the full 20-year baseline, then nearest-neighbour-snapping onto that tiled
timestamp array — instead of cropping once to the original ~5-8yr baseline
and discarding the rest of the simulated curve.

This is the test Franz asked for: extend the intrinsic simulation (already
done in stage1) but reuse the SAME real sampling pattern repeated over the
longer baseline, so any turnover in the structure function can be checked
for whether it stays put (real break) or keeps moving out / disappears
(cadence/baseline artifact) as more of the light curve becomes visible.

Difference from stage2_crop_lc.py: only `load_original_times`'s target-find
call and `original_file_for`'s glob are fixed to match stage1's proven-working
versions, and one step (`repeat_cadence`) is inserted before the cKDTree
query. Everything else (per-source parallelism, sentinel scheme, output
schema) is unchanged.

Usage:
    mpirun -n <nranks> python stage2_crop_lc_20yr.py

Input  : ~/results/20yr_lc/full_lc/sim_{ra}_{dec}_z{band}_rank*_full.parquet (stage1)
         + original ZTF parquet (for observation times)
Output : ~/results/20yr_lc/cropped_lc/sim_{ra}_{dec}_z{band}_cropped.parquet
Sentinel: ~/results/20yr_lc/cropped_lc/sim_{ra}_{dec}_z{band}.stage2.done
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
DATA_PATH   = os.environ["HOME"] + "/BAT_results/"
FULL_PATH   = os.environ["HOME"] + "/results/20yr_lc/full_lc/"
OUTPUT_PATH = os.environ["HOME"] + "/results/20yr_lc/cropped_lc/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_original_lc(ra, dec, path):
    """Original observation times AFTER the same masking stage 1 used.

    Must byte-match stage 1's load_original_lc filtering so the tiled crop
    reuses the exact same real-cadence pattern the DRW fit was based on.
    Returns time_orig (relative, starts at 0) or None.
    """
    ztf   = pd.read_parquet(path)
    coord = SkyCoord(ra, dec, unit="deg")
    tgt_obj_idx = _find_target_obj(Path(path), coord)
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

    # time_arr = ztf1.OBSMJD.values.copy()
    # time_arr.sort()
    # time_arr -= time_arr[0]
    # return time_arr
    return ztf1


def repeat_cadence(time_orig, target_dur):
    """Tile the real observed cadence pattern back-to-back until it spans
    target_dur days, then trim anything past target_dur.

    time_orig must be relative and start at 0 (as returned above). Each
    repeat is shifted by the full baseline (time_orig.max()), so repeat
    i+1's first epoch sits immediately after repeat i's last epoch — a
    literal repeat of the cadence, with no inserted season gap at the seam.
    Exact-duplicate boundary points are removed with np.unique (which also
    guarantees a sorted, monotonic array for the cKDTree query below).
    """
    baseline = time_orig.max()
    if baseline <= 0:
        raise ValueError("time_orig spans zero days, cannot tile")
    n_reps = int(np.ceil(target_dur / baseline)) + 1
    tiled  = np.concatenate([time_orig + i * baseline for i in range(n_reps)])
    tiled  = np.unique(tiled[tiled <= target_dur])
    return tiled


_original_file_index = None

def _build_original_file_index():
    """(ra, dec, band) -> path, built once by globbing DATA_PATH a single
    time. ra/dec are floats parsed from the filename, so lookups must key
    on the same parsed floats rather than re-formatted strings (fixed
    decimal places in the filenames, e.g. "45.100000", don't round-trip
    through float -> str)."""
    idx = {}
    for f in glob.glob(DATA_PATH + "*z[gri]_merged.parquet"):
        try:
            r, d, b = radec_filename(f, band=True)
        except ValueError:
            continue
        idx[(r, d, b)] = f
    return idx


def original_file_for(ra, dec, band):
    """Locate the source's original ZTF parquet. Matches stage1's glob."""
    global _original_file_index
    if _original_file_index is None:
        _original_file_index = _build_original_file_index()
    return _original_file_index.get((ra, dec, band))


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
    """Crop every sim of one source onto the repeated cadence, write one parquet."""
    full = load_full_source(ra, dec, band)
    if full is None:
        print(f"[rank {rank}] no stage-1 data for {ra}_{dec}_z{band}", flush=True)
        return False

    orig_path = original_file_for(ra, dec, band)
    if orig_path is None:
        print(f"[rank {rank}] no original file for {ra}_{dec}_z{band}", flush=True)
        return False
    orig_lc = load_original_lc(ra, dec, orig_path)
    if orig_lc is None:
        print(f"[rank {rank}] no usable original times for {ra}_{dec}_z{band}", flush=True)
        return False
    time_orig = orig_lc.OBSMJD.values.copy()
    time_orig.sort()
    time_orig -= time_orig[0]

    # even-cadence grid: identical across sims — take it from the first sim
    idx0   = full["object_index"].min()
    t_even = full[full.object_index == idx0].sort_values("time")["time"].values

    # tile the real cadence over the full simulated baseline (~20 yr), then
    # nearest-neighbour-map the tiled timestamps onto the even-cadence grid
    time_rep = repeat_cadence(time_orig, t_even.max())
    tree     = cKDTree(t_even.reshape(-1, 1))
    _, cropi = tree.query(time_rep.reshape(-1, 1))
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
    print(f"[rank {rank}] {ra}_{dec}_z{band}: repeated cadence "
          f"{len(time_orig)}->{len(time_rep)} epochs/rep, snapped from "
          f"{len(t_even)}-pt grid x {full['object_index'].nunique()} sims "
          f"-> {path}", flush=True)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main — static round-robin source assignment across ranks
# ─────────────────────────────────────────────────────────────────────────────

def main():
    job_start = time.time()

    sentinels = sorted(glob.glob(FULL_PATH + "*.stage1.done"))
    sources = []
    for p in sentinels:
        m = re.search(r"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.stage1\.done",
                      os.path.basename(p))
        if m:
            sources.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    sources.sort()

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
        print(f"[rank 0] stage 2 (20yr) walltime {time.time()-job_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
