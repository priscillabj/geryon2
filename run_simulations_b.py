"""
run_simulations.py

Usage on the cluster:
    mpirun -n <nranks> python run_simulations.py

Requirements:
    mpi4py, eztao, numpy, pandas, scipy, linmix, celerite

Parallelisation strategy
────────────────────────
All ranks collaborate on one source at a time:

  1. Rank 0 builds and broadcasts the list of source files.
  2. For each source, rank 0 alone loads the light curve and fits DRW once,
     then broadcasts a small `prep` dict (time array + a few scalars). This
     avoids 80 ranks redundantly fitting identical data and hammering the
     shared filesystem with simultaneous reads of the same parquet.
  3. Every rank generates its own slice of N_SIMS / nrank simulations from
     the broadcast DRW parameters and runs SF + LinMix on it.
  4. comm.gather() collects all slices to rank 0 — MPI-synchronised and
     guaranteed, unlike filesystem globs which can miss files on shared
     storage under high concurrency.
  5. Rank 0 merges the slices and writes the result to the main checkpoint
     pickle before moving to the next source.
  6. The job can be killed and restarted at any point: completed sources
     (keyed by (RA, DEC, band)) are skipped on restart.
"""

import os
import re
import glob
import time
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from mpi4py import MPI

from eztao.ts import drw_fit
from eztao.carma import DRW_term
from eztao.ts import gpSimFull

from newSF import SF_wnoise, SF_linmix
from VarTools import _find_target_obj

# ── configuration ─────────────────────────────────────────────────────────────
# N_SIMS      = 10_000
N_SIMS      = 120
DATA_PATH   = os.environ["HOME"] + "/BAT_results/"
OUTPUT_PATH = os.environ["HOME"] + "/results/partials/"
OUTPUT_PKL  = OUTPUT_PATH + f"new{N_SIMS}simfit_results_b.pkl"

# Set to True to save simulated LCs as parquet files (one per rank per source).
# Useful for testing and debugging — not recommended for large N_SIMS runs.
# Run merge_simlc.py afterwards to combine rank files into one file per source.
SAVE_LC     = True

FIT_KEYS = [
    "_A_1", "_A_365_spl", "_A_maxerr_spl", "_A_minerr_spl",
    "_gamma_spl", "_gamma_maxerr_spl", "_gamma_minerr_spl"
]

# ── MPI setup ─────────────────────────────────────────────────────────────────
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def radec_filename(filename):
    match  = re.search(r"(\d+\.\d+)_([-+]?\d+\.\d+)", filename)
    match2 = re.search(r"_z(\w)_", filename)
    band   = match2.group(1)
    if match:
        return float(match.group(1)), float(match.group(2)), band
    raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")


def load_original_lc(ra, dec, path):
    """Load the original ZTF light curve for one source.

    Returns (time_arr, mag, mag_err) or None if no usable target is found.
    """
    ztf = pd.read_parquet(path)
    tgt_obj_idx = _find_target_obj(ra, dec, ztf)
    if tgt_obj_idx is None:
        print(f"no target found in file, skipping")
        return None

    # obj_id = ztf.loc[tgt_obj_idx, "object_index"]
    # ztf1   = ztf[ztf["object_index"] == obj_id]
    ztf1 = ztf[ztf['object_index'] == tgt_obj_idx]

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
        print("empty target")
        return None

    time_arr = ztf1.OBSMJD.values.copy()
    time_arr -= time_arr[0]
    mag      = ztf1["MAG_4_TOT_AB"].values
    mag_err  = ztf1["MERR_4_TOT_AB"].values

    return time_arr, mag, mag_err


def build_simcad(t_even, y_i, yerr_i, time_orig, tree):
    """Nearest-neighbour cadence matching (unchanged from original)."""
    sim    = pd.DataFrame({"time": t_even, "mag": y_i, "err": yerr_i})
    _, idx = tree.query(time_orig.reshape(-1, 1))
    simcad = pd.DataFrame({
        "time": t_even[idx],
        "mag" : y_i[idx],
        "err" : yerr_i[idx],
    })
    return sim, simcad


def save_lc_parquet(ra, dec, band, rows_daily, rows_orgcad):
    """
    Write one parquet file per cadence per rank for this source.
    Each row is one epoch of one simulation, identified by object_index.
    Called only when SAVE_LC=True.

    Columns: object_index, time, mag, err, valid_fit, fit_fail_reason
    """
    if rows_daily:
        path = OUTPUT_PATH + f"sim_{ra}_{dec}_rank{rank}_{band}_daily.parquet"
        pd.concat(rows_daily, ignore_index=True).to_parquet(path, index=False)
        print(f"[rank {rank}] saved {path}", flush=True)

    if rows_orgcad:
        path = OUTPUT_PATH + f"sim_{ra}_{dec}_rank{rank}_{band}_orgcad.parquet"
        pd.concat(rows_orgcad, ignore_index=True).to_parquet(path, index=False)
        print(f"[rank {rank}] saved {path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint():
    """Return existing results dict keyed by (ra, dec, band), or empty dict."""
    if os.path.exists(OUTPUT_PKL):
        with open(OUTPUT_PKL, "rb") as f:
            existing = pickle.load(f)
        return {(r["RA"], r["DEC"], r["ref_band"]): r for r in existing}
    return {}


def save_checkpoint(results_dict):
    """Atomically overwrite the main pickle."""
    tmp = OUTPUT_PKL + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(list(results_dict.values()), f)
    os.replace(tmp, OUTPUT_PKL)          # atomic on POSIX shared filesystems


def is_complete(result):
    return result.get("n_sims", 0) >= N_SIMS


def merge_and_save(ra, dec, band, time_arr, dur, all_slices):
    """
    Rank 0 merges gathered slices from all ranks and saves to the checkpoint.
    all_slices: list of (dailycad, orgcad) tuples, one per rank.
    """
    dailycad_all = []
    orgcad_all   = []
    for dailycad, orgcad in all_slices:
        dailycad_all.extend(dailycad)
        orgcad_all.extend(orgcad)

    result = {
        "RA"          : ra,
        "DEC"         : dec,
        "ref_band"    : band,
        "n_epochs"    : len(time_arr),
        "length"      : np.float32(dur),
        "mean_cad"    : np.float32(np.mean(np.diff(time_arr))),
        "n_sims"      : len(dailycad_all),
        "1day_gamma"  : np.array(dailycad_all, dtype=np.float32),
        "orgcad_gamma": np.array(orgcad_all,   dtype=np.float32),
    }

    done = load_checkpoint()
    done[(ra, dec, band)] = result          # 3-tuple key, matches load_checkpoint
    save_checkpoint(done)
    print(f"[rank 0] merged {len(all_slices)} rank slices → "
          f"{len(dailycad_all)} valid sims for RA={ra} DEC={dec}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Source preparation (rank 0 only) and per-rank simulation slice (all ranks)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_source(filename):
    """
    Rank 0 only: load the light curve and fit DRW *once*.

    Returns a small dict suitable for broadcasting, or None if the source has
    no usable target (so all ranks can skip it together).
    """
    ra, dec, band = radec_filename(filename)

    loaded = load_original_lc(ra, dec, filename)
    if loaded is None:
        return None
    time_arr, mag, mag_err = loaded

    best_fit = drw_fit(time_arr, mag, mag_err)
    print(f"[rank 0] RA={ra} DEC={dec} DRW fit done: {best_fit}", flush=True)

    return {
        "ra"          : ra,
        "dec"         : dec,
        "band"        : band,
        "time_arr"    : time_arr,
        "dur"         : float(time_arr[-1]),
        "N"           : len(time_arr),
        "mag_med"     : float(np.median(mag)),
        "mag_err_med" : float(np.median(mag_err)),
        "best_fit"    : best_fit,
    }


def run_slice(prep):
    """
    Every rank runs this for the current source, using the broadcast `prep`.

    Each rank is responsible for a contiguous slice of the N_SIMS simulations:
        slice_size = N_SIMS // nrank  (+1 for the first `remainder` ranks)

    gpSimFull is called with only the slice size, so memory scales with
    N_SIMS / nrank rather than N_SIMS.

    Returns (dailycad, orgcad) — lists of fitted gamma values for this slice.
    """
    ra, dec, band = prep["ra"], prep["dec"], prep["band"]
    time_arr      = prep["time_arr"]
    mag_med       = prep["mag_med"]
    dur, N        = prep["dur"], prep["N"]

    SNR        = prep["best_fit"][0] / prep["mag_err_med"]
    DRW_kernel = DRW_term(*np.log(prep["best_fit"]))

    # ── assign this rank's simulation slice ──────────────────────────────────
    base, remainder = divmod(N_SIMS, nrank)
    slice_size      = base + (1 if rank < remainder else 0)

    if slice_size == 0:
        # more ranks than sims — this rank has nothing to do
        return [], []

    # ── generate only this rank's LCs ────────────────────────────────────────
    # Use rank as part of the seed so each rank produces different LCs
    t_batch, y_batch, yerr_batch = gpSimFull(
        DRW_kernel, SNR, dur, N, nLC=slice_size,
        log_flux=True, lc_seed=rank
    )
    t_even = t_batch[0] if slice_size > 1 else t_batch

    # Ensure y_batch / yerr_batch are always 2-D (gpSimFull returns 1-D if nLC=1)
    if slice_size == 1:
        y_batch    = y_batch[np.newaxis, :]
        yerr_batch = yerr_batch[np.newaxis, :]

    # ── run SF + LinMix on each sim in the slice ──────────────────────────────
    dailycad = []
    orgcad   = []
    if SAVE_LC:
        rows_daily  = []
        rows_orgcad = []

    # global sim index offset for this rank, so object_index is unique across ranks
    sim_offset = rank * base + min(rank, remainder)

    # build cKDTree once — t_even is identical for all sims in this slice
    tree  = cKDTree(t_even.reshape(-1, 1))
    width = len(str(slice_size))            # fixed width for timing print

    source_start = time.time()

    for i in range(slice_size):
        sim_start  = time.time()
        object_idx = sim_offset + i

        y_i    = y_batch[i] + mag_med
        yerr_i = yerr_batch[i]

        sim, simcad = build_simcad(t_even, y_i, yerr_i, time_arr, tree)

        sf_full_dict = SF_wnoise(sim["mag"],    sim["time"],    sim["err"], color="#436BAD")
                                 #save_plt=True,path=OUTPUT_PATH+f'{ra}_{dec}/z{band}_sci_merged_{object_idx}_')
        sf_cad_dict  = SF_wnoise(simcad["mag"], simcad["time"], simcad["err"], color="orange")
                                 #save_plt=True,path=OUTPUT_PATH+f'{ra}_{dec}/z{band}_sci_merged_{object_idx}_cad')

        fit_full = SF_linmix(sf_full_dict[0])
        fit_cad  = SF_linmix(sf_cad_dict[0])

        # ── determine fit validity and reason ────────────────────────────────
        valid       = False
        fail_reason = ""
        if fit_full is None:
            fail_reason = "SF_linmix returned None (daily)"
        elif fit_cad is None:
            fail_reason = "SF_linmix returned None (orgcad)"
        else:
            try:
                dailycad.append(fit_full["_gamma_spl"])
                orgcad.append(fit_cad["_gamma_spl"])
                valid = True
            except Exception as e:
                fail_reason = f"key error: {e}"
                print(f"[rank {rank}]   sim {i} key error: {e}", flush=True)

        # ── optionally collect LC rows ────────────────────────────────────────
        if SAVE_LC:
            fit_full_vals = fit_full if fit_full is not None else {k: np.nan for k in FIT_KEYS}
            fit_cad_vals  = fit_cad  if fit_cad  is not None else {k: np.nan for k in FIT_KEYS}

            sim["object_index"]    = object_idx
            sim["valid_fit"]       = valid
            sim["fit_fail_reason"] = fail_reason
            sim = sim.assign(**fit_full_vals)
            rows_daily.append(sim)

            simcad["object_index"]    = object_idx
            simcad["valid_fit"]       = valid
            simcad["fit_fail_reason"] = fail_reason
            simcad = simcad.assign(**fit_cad_vals)
            rows_orgcad.append(simcad)

        # ── timing report ─────────────────────────────────────────────────────
        elapsed      = time.time() - sim_start
        total_so_far = time.time() - source_start
        avg_sim_time = total_so_far / (i + 1)
        remaining    = avg_sim_time * (slice_size - (i + 1))

        print(
            f"[rank {rank}]   sim {i+1:{width}}/{slice_size} | "
            f"this: {elapsed:5.1f}s | avg: {avg_sim_time:5.1f}s | "
            f"elapsed: {total_so_far:6.1f}s | ETA: {remaining:6.1f}s",
            flush=True
        )

    # ── save LC parquets if requested ─────────────────────────────────────────
    if SAVE_LC:
        save_lc_parquet(ra, dec, band, rows_daily, rows_orgcad)

    print(f"[rank {rank}] slice done: {len(dailycad)} valid sims "
          f"for RA={ra} DEC={dec}", flush=True)

    return dailycad, orgcad


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    job_start = time.time()

    # ── rank 0: build the work list ──────────────────────────────────────────
    if rank == 0:
        all_files = sorted(glob.glob(DATA_PATH + "*sci_merged.parquet"))
        done      = load_checkpoint()
        todo      = []
        skipped   = 0

        for f in all_files[27:30]:
            try:
                print(f)
                ra, dec, band = radec_filename(f)
                print(band)
            except ValueError:
                continue
            key = (ra, dec, band)               # 3-tuple, matches load_checkpoint
            if key in done and is_complete(done[key]):
                skipped += 1
            else:
                todo.append(f)

        print(f"[rank 0] {len(all_files)} total | "
              f"{skipped} complete | {len(todo)} to process", flush=True)
    else:
        todo = None

    # ── broadcast todo list ──────────────────────────────────────────────────
    todo = comm.bcast(todo, root=0)

    # ── process sources one at a time, all ranks collaborating on each ───────
    for f in todo:
        comm.Barrier()          # all ranks start each source together
        t0 = time.time()

        # ── rank 0 loads + fits once, then broadcasts the small prep dict ────
        prep = prepare_source(f) if rank == 0 else None
        prep = comm.bcast(prep, root=0)

        if prep is None:                    # no usable target — skip on all ranks
            if rank == 0:
                print(f"[rank 0] skip {f} (no data)", flush=True)
            continue

        # ── every rank runs its slice from the shared DRW parameters ─────────
        try:
            dailycad, orgcad = run_slice(prep)
        except Exception as e:
            print(f"[rank {rank}] ERROR on {f}: {e}", flush=True)
            dailycad, orgcad = [], []       # failed ranks still contribute empty

        # ── gather all slices to rank 0 (MPI-synchronised, no filesystem race) ─
        all_slices = comm.gather((dailycad, orgcad), root=0)

        # ── rank 0 merges and saves immediately ──────────────────────────────
        if rank == 0:
            try:
                merge_and_save(prep["ra"], prep["dec"], prep["band"],
                               prep["time_arr"], prep["dur"], all_slices)
                print(f"[rank 0] source done in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[rank 0] merge ERROR for {f}: {e}", flush=True)

    comm.Barrier()

    if rank == 0:
        done  = load_checkpoint()
        total = time.time() - job_start
        print(f"[rank 0] all done. {len(done)} sources in {OUTPUT_PKL}.", flush=True)
        print(f"[rank 0] total walltime: {total:04.1f}s", flush=True)


if __name__ == "__main__":
    main()