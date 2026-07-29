# """
# run_simulations.py

# Usage on the cluster:
#     mpirun -n <nranks> python run_simulations.py

# Requirements:
#     mpi4py, eztao, numpy, pandas, scipy, linmix, celerite

# Parallelisation strategy
# ────────────────────────
# The old code assigned whole sources to ranks, so each rank ran all N_SIMS
# simulations alone.  With 5 sources × 10 000 sims and 120 ranks that left
# most of the parallelism on the table.

# The new strategy parallelises *across simulations* for every source:

#   1. Rank 0 broadcasts the list of source files.
#   2. All ranks collaborate on one source at a time (barrier-synchronised).
#      - Every rank independently fits DRW and generates its own sim slice
#        (gpSimFull is cheap and stateless, so duplicating the DRW fit across
#        ranks is far cheaper than a broadcast of 10 000 LCs).
#      - Each rank runs SF + LinMix on its slice and immediately writes a
#        small partial pickle:  partials/<ra>_<dec>_rank<r>.pkl
#   3. Once all ranks finish a source, rank 0 merges the partial files,
#      writes the final combined result to simfit_results.pkl, and deletes
#      the partial files.
#   4. The job can be killed and restarted at any point: completed sources are
#      skipped, and partial files from an interrupted source are cleaned up and
#      re-run.
# """

# import os
# import re
# import glob
# import time
# import pickle
# import numpy as np
# import pandas as pd
# from scipy.spatial import cKDTree
# from mpi4py import MPI

# from eztao.ts import drw_fit
# from eztao.carma import DRW_term
# from eztao.ts import gpSimFull
# from celerite import GP

# from SF import SF_wnoise, SF_linmix

# # ── configuration ─────────────────────────────────────────────────────────────
# N_SIMS      = 1200
# DATA_PATH   = os.environ["HOME"] + "/LC/"
# OUTPUT_PKL  = "results/simfit_results.pkl"
# PARTIAL_DIR = "results/partials/"          # one file per rank per source

# # ── MPI setup ─────────────────────────────────────────────────────────────────
# comm  = MPI.COMM_WORLD
# rank  = comm.Get_rank()
# nrank = comm.Get_size()


# # ─────────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def radec_filename(filename):
#     match = re.search(r"lc_(\d+\.\d+)_(-?\d+\.\d+)", filename)
#     if match:
#         return float(match.group(1)), float(match.group(2))
#     raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")


# def load_original_lc(ra, dec):
#     """Load the original ZTF g-band light curve for one source."""
#     path = DATA_PATH + f"corrlc_{ra}_{dec}.csv"
#     ztf  = pd.read_csv(path)
#     ztf  = ztf[ztf["filter"] == "ZTF_g"]

#     time_arr = ztf.mjd.values.copy()
#     time_arr -= time_arr[0]
#     mag      = ztf.magtot_clr_corr.values
#     mag_err  = ztf.magunc_clr_corr.values

#     flux     = ztf.flux_tot_clr_uJy_corr.values
#     flux_err = ztf.fluxunc_tot_clr_uJy_corr_alt2.values
#     snr_med  = np.nanmedian(flux / flux_err)
#     SNR      = 20 if np.isnan(snr_med) else min(snr_med, 20)

#     return time_arr, mag, mag_err, SNR


# def build_simcad(t_even, y_i, yerr_i, time_orig):
#     """Nearest-neighbour cadence matching (unchanged from original)."""
#     sim    = pd.DataFrame({"time": t_even, "mag": y_i, "err": yerr_i})
#     tree   = cKDTree(t_even.reshape(-1, 1))
#     _, idx = tree.query(time_orig.reshape(-1, 1))
#     simcad = pd.DataFrame({
#         "time": t_even[idx],
#         "mag" : y_i[idx],
#         "err" : yerr_i[idx],
#     })
#     return sim, simcad


# # ─────────────────────────────────────────────────────────────────────────────
# # Checkpoint helpers  (rank 0 only)
# # ─────────────────────────────────────────────────────────────────────────────

# def load_checkpoint():
#     """Return existing results dict keyed by (ra, dec), or empty dict."""
#     if os.path.exists(OUTPUT_PKL):
#         with open(OUTPUT_PKL, "rb") as f:
#             existing = pickle.load(f)
#         return {(r["RA"], r["DEC"]): r for r in existing}
#     return {}


# def save_checkpoint(results_dict):
#     """Atomically overwrite the main pickle."""
#     tmp = OUTPUT_PKL + ".tmp"
#     with open(tmp, "wb") as f:
#         pickle.dump(list(results_dict.values()), f)
#     os.replace(tmp, OUTPUT_PKL)          # atomic on POSIX shared filesystems


# def is_complete(result):
#     return result.get("n_sims", 0) >= N_SIMS


# def partial_path(ra, dec):
#     return os.path.join(PARTIAL_DIR, f"{ra}_{dec}_rank{rank}.pkl")


# def save_partial(ra, dec, dailycad, orgcad):
#     """Each rank saves its own slice immediately after finishing a source."""
#     path = partial_path(ra, dec)
#     with open(path, "wb") as f:
#         pickle.dump({"RA": ra, "DEC": dec,
#                      "1day_gamma": dailycad, "orgcad_gamma": orgcad}, f)


# def merge_partials(ra, dec, time_arr, dur):
#     """
#     Rank 0 reads all rank partial files for this source, merges them into
#     the final result dict, saves to the main checkpoint, and removes the
#     partial files.
#     """
#     pattern = os.path.join(PARTIAL_DIR, f"{ra}_{dec}_rank*.pkl")
#     files   = glob.glob(pattern)

#     dailycad_all = []
#     orgcad_all   = []
#     for fp in files:
#         with open(fp, "rb") as f:
#             part = pickle.load(f)
#         dailycad_all.extend(part["1day_gamma"])
#         orgcad_all.extend(part["orgcad_gamma"])

#     result = {
#         "RA"          : ra,
#         "DEC"         : dec,
#         "ref_band"    : "g",
#         "n_epochs"    : len(time_arr),
#         "length"      : dur,
#         "mean_cad"    : float(np.mean(np.diff(time_arr))),
#         "n_sims"      : len(dailycad_all),
#         "1day_gamma"  : dailycad_all,
#         "orgcad_gamma": orgcad_all,
#     }

#     done = load_checkpoint()
#     done[(ra, dec)] = result
#     save_checkpoint(done)
#     print(f"[rank 0] merged {len(files)} partial files → "
#           f"{len(dailycad_all)} valid sims for RA={ra} DEC={dec}", flush=True)

#     for fp in files:
#         os.remove(fp)

#     return result


# def cleanup_stale_partials(ra, dec):
#     """Remove leftover partial files from a previously interrupted run."""
#     for fp in glob.glob(os.path.join(PARTIAL_DIR, f"{ra}_{dec}_rank*.pkl")):
#         os.remove(fp)
#         print(f"[rank 0] removed stale partial {fp}", flush=True)


# # ─────────────────────────────────────────────────────────────────────────────
# # Per-rank simulation slice
# # ─────────────────────────────────────────────────────────────────────────────

# def process_slice(filename):
#     """
#     Every rank runs this for the current source.

#     Each rank is responsible for a contiguous slice of the N_SIMS simulations:
#         my_start = rank * slice_size
#         my_end   = my_start + slice_size   (last rank may get fewer)

#     gpSimFull is called with only the slice size, so memory scales with
#     N_SIMS / nrank rather than N_SIMS.
#     """
#     ra, dec = radec_filename(filename)

#     # ── load LC (all ranks read independently — fast, file is small) ────────
#     time_arr, mag, mag_err, SNR = load_original_lc(ra, dec)
#     N       = len(time_arr)
#     dur     = time_arr[-1]
#     mag_med = np.median(mag)

#     # ── DRW fit (all ranks do this independently — cheaper than broadcast) ──
#     best_fit   = drw_fit(time_arr, mag, mag_err)
#     DRW_kernel = DRW_term(*np.log(best_fit))
#     print(f"[rank {rank}] RA={ra} DEC={dec} DRW fit done: {best_fit}", flush=True)

#     # ── assign this rank's simulation slice ──────────────────────────────────
#     base       = N_SIMS // nrank
#     remainder  = N_SIMS  % nrank
#     slice_size = base + (1 if rank < remainder else 0)

#     if slice_size == 0:
#         # more ranks than sims — this rank has nothing to do
#         save_partial(ra, dec, [], [])
#         return time_arr, dur

#     # ── generate only this rank's LCs ────────────────────────────────────────
#     # Use rank as part of the seed so each rank produces different LCs
#     t_batch, y_batch, yerr_batch = gpSimFull(
#         DRW_kernel, SNR, dur, N, nLC=slice_size,
#         log_flux=False, lc_seed=rank
#     )
#     t_even = t_batch[0] if slice_size > 1 else t_batch

#     # Ensure y_batch / yerr_batch are always 2-D (gpSimFull returns 1-D if nLC=1)
#     if slice_size == 1:
#         y_batch    = y_batch[np.newaxis, :]
#         yerr_batch = yerr_batch[np.newaxis, :]

#     # ── run SF + LinMix on each sim in the slice ──────────────────────────────
#     dailycad = []
#     orgcad   = []

#     sim_times    = []
#     source_start = time.time()

#     for i in range(slice_size):
#         sim_start = time.time()

#         y_i    = y_batch[i] + mag_med
#         yerr_i = yerr_batch[i]

#         sim, simcad = build_simcad(t_even, y_i, yerr_i, time_arr)

#         sf_full_dict = SF_wnoise(sim["mag"],    sim["time"],    sim["err"],
#                                  plot=False, color="#436BAD")
#         sf_cad_dict  = SF_wnoise(simcad["mag"], simcad["time"], simcad["err"],
#                                  plot=False, color="orange")

#         fit_full = SF_linmix(sf_full_dict[0])
#         fit_cad  = SF_linmix(sf_cad_dict[0])

#         if fit_cad is not None and fit_full is not None:
#             try:
#                 dailycad.append(fit_full["_gamma"])
#                 orgcad.append(fit_cad["_gamma"])
#             except Exception as e:
#                 print(f"[rank {rank}]   sim {i} key error: {e}", flush=True)

#         # ── timing report ─────────────────────────────────────────────────────
#         elapsed      = time.time() - sim_start
#         sim_times.append(elapsed)
#         avg_sim_time = np.mean(sim_times)
#         remaining    = avg_sim_time * (slice_size - (i + 1))
#         total_so_far = time.time() - source_start
#         width        = len(str(slice_size))

#         print(
#             f"[rank {rank}]   sim {i+1:{width}}/{slice_size} | "
#             f"this: {elapsed:5.1f}s | avg: {avg_sim_time:5.1f}s | "
#             f"elapsed: {total_so_far:6.1f}s | ETA: {remaining:6.1f}s",
#             flush=True
#         )

#     # ── save this rank's results immediately ─────────────────────────────────
#     save_partial(ra, dec, dailycad, orgcad)
#     print(f"[rank {rank}] saved partial: {len(dailycad)} valid sims "
#           f"for RA={ra} DEC={dec}", flush=True)

#     return time_arr, dur


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     # ── rank 0: create partial dir and build the work list ───────────────────
#     if rank == 0:
#         os.makedirs(PARTIAL_DIR, exist_ok=True)
#         #all_files = sorted(glob.glob(DATA_PATH + "calibration_sources/PS1/*/lc_*.parquet"))
#         all_files = sorted(glob.glob(DATA_PATH + "corrlc_*.csv"))
#         done      = load_checkpoint()
#         todo      = []
#         skipped   = 0

#         for f in all_files[:5]:
#             print(f)
#             try:
#                 ra, dec = radec_filename(f)
#             except ValueError:
#                 print('radec_filename failed')
#                 continue
#             key = (ra, dec)
#             if key in done and is_complete(done[key]):
#                 skipped += 1
#             else:
#                 # clean up any leftover partial files from a previous run
#                 cleanup_stale_partials(ra, dec)
#                 todo.append(f)

#         print(f"[rank 0] {len(all_files)} total | "
#               f"{skipped} complete | {len(todo)} to process", flush=True)
#     else:
#         todo = None

#     # ── broadcast todo list ──────────────────────────────────────────────────
#     todo = comm.bcast(todo, root=0)

#     # ── process sources one at a time, all ranks collaborating on each ───────
#     for f in todo:
#         comm.Barrier()          # all ranks start each source together

#         t0 = time.time()
#         try:
#             time_arr, dur = process_slice(f)
#         except Exception as e:
#             print(f"[rank {rank}] ERROR on {f}: {e}", flush=True)
#             # write an empty partial so rank 0 doesn't wait forever
#             try:
#                 ra, dec = radec_filename(f)
#                 save_partial(ra, dec, [], [])
#             except Exception:
#                 pass

#         comm.Barrier()          # wait for all ranks to finish + save partials

#         # ── rank 0 merges and saves the final result for this source ─────────
#         if rank == 0:
#             try:
#                 ra, dec  = radec_filename(f)
#                 merge_partials(ra, dec, time_arr, dur)
#                 print(f"[rank 0] source done in {time.time()-t0:.1f}s", flush=True)
#             except Exception as e:
#                 print(f"[rank 0] merge ERROR for {f}: {e}", flush=True)

#     comm.Barrier()
#     if rank == 0:
#         done = load_checkpoint()
#         print(f"[rank 0] all done. {len(done)} sources in {OUTPUT_PKL}.", flush=True)


# if __name__ == "__main__":
#     main()


"""
run_simulations.py

Usage on the cluster:
    mpirun -n <nranks> python run_simulations.py

Requirements:
    mpi4py, eztao, numpy, pandas, scipy, linmix, celerite

Parallelisation strategy
────────────────────────
All ranks collaborate on one source at a time:

  1. Rank 0 broadcasts the list of source files.
  2. Every rank independently fits DRW and runs gpSimFull on its own slice
     of N_SIMS / nrank simulations.
  3. comm.gather() collects all slices to rank 0 — this is MPI-synchronised
     and guaranteed, unlike filesystem globs which can miss files on shared
     storage under high concurrency.
  4. Rank 0 merges the slices and immediately writes the result to the main
     checkpoint pickle before moving to the next source.
"""

import os
import re
import sys
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
# from celerite import GP

# import pyarrow as pa
import pyarrow.parquet as pq

from newSF import SF_wnoise, SF_linmix
from VarTools import _find_target_obj

# ── configuration ─────────────────────────────────────────────────────────────
# N_SIMS      = 10_000
N_SIMS      = 120
# DATA_PATH   = os.environ["HOME"] + "/LC/"
DATA_PATH   = os.environ["HOME"] + "/BAT_results/"
OUTPUT_PATH = os.environ["HOME"] + f"/results/partials/"
OUTPUT_PKL  = OUTPUT_PATH + f"new{N_SIMS}simfit_results.pkl"

# Set to True to save simulated LCs as parquet files (one per rank per source).
# Useful for testing and debugging — not recommended for large N_SIMS runs.
# Run merge_simlc.py afterwards to combine rank files into one file per source.
SAVE_LC     = True
# LC_PATH     = DATA_PATH + "sim_lc/"

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
    # match = re.search(r"corrlc_(\d+\.\d+)_(-?\d+\.\d+)", filename)
    match = re.search(r"(\d+\.\d+)_([-+]?\d+\.\d+)", filename)
    match2 = re.search(r'_z(\w)_', filename)
    # print(file)
    band  = match2.group(1)
    if match:
        return float(match.group(1)), float(match.group(2)), band
    raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")


# def load_original_lc(ra, dec, band='g'):
# def load_original_lc(ra, dec, band):
def load_original_lc(ra, dec, path):
    """Load the original ZTF g-band light curve for one source."""
    # path = DATA_PATH + f"corrlc_{ra}_{dec}.csv"
    # path = DATA_PATH + f"{ra}_{dec}_z{band}_merged.parquet"
    # ztf  = pd.read_csv(path)
    ztf = pd.read_parquet(path)
    tgt_obj_idx = _find_target_obj(ra, dec, ztf)
    if tgt_obj_idx is None:
        print(f'no target found in file, skipping')
        return None
    # ztf  = ztf[ztf["filter"] == f"ZTF_{band}"]

    # time_arr = ztf.mjd.values.copy()
    # time_arr = ztf.OBSMJD.values.copy()
    # time_arr -= time_arr[0]
    # # mag      = ztf.magtot_clr_corr.values
    # # mag_err  = ztf.magunc_clr_corr.values
    # mag       = ztf['MAG_4_TOT_AB'].values
    # mag_err   = ztf['MERR_4_TOT_AB'].values

    obj_id = ztf.loc[tgt_obj_idx, 'object_index']
    ztf1 = ztf[ztf['object_index'] == obj_id]

    mask = ztf1['MERR_4_TOT_AB'] < 0.5

    if 'qid' in ztf1.columns:
        qid    = ztf1['qid'].mode().iloc[0]
        ccdid  = ztf1['ccdid'].mode().iloc[0]
        field  = ztf1['field'].mode().iloc[0]
        mask &= (
        (ztf1['qid']   == qid)  &
        (ztf1['ccdid'] == ccdid) &
        (ztf1['field'] == field)
    )

    ztf1 = ztf1[mask]

    if len(ztf1) == 0:
        print('empty target')
        return None
    
    time_arr = ztf1.OBSMJD.values.copy()
    time_arr -= time_arr[0]
    # mag      = ztf.magtot_clr_corr.values
    # mag_err  = ztf.magunc_clr_corr.values
    mag       = ztf1['MAG_4_TOT_AB'].values
    mag_err   = ztf1['MERR_4_TOT_AB'].values


    # flux     = ztf.flux_tot_clr_uJy_corr.values
    # flux_err = ztf.fluxunc_tot_clr_uJy_corr_alt2.values
    # snr_med  = np.nanmedian(flux / flux_err)
    # SNR      = 20 if np.isnan(snr_med) else min(snr_med, 20)

    return time_arr, mag, mag_err#, SNR


def build_simcad(t_even, y_i, yerr_i, time_orig,tree):
    """Nearest-neighbour cadence matching (unchanged from original)."""
    sim    = pd.DataFrame({"time": t_even, "mag": y_i, "err": yerr_i})
    # tree   = cKDTree(t_even.reshape(-1, 1))
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
    # os.makedirs(LC_PATH, exist_ok=True)

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
    """Return existing results dict keyed by (ra, dec), or empty dict."""
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
    done[(ra, dec)] = result
    save_checkpoint(done)
    print(f"[rank 0] merged {len(all_slices)} rank slices → "
          f"{len(dailycad_all)} valid sims for RA={ra} DEC={dec}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-rank simulation slice
# ─────────────────────────────────────────────────────────────────────────────

def process_slice(filename):
    """
    Every rank runs this for the current source.

    Each rank is responsible for a contiguous slice of the N_SIMS simulations:
        my_start = rank * slice_size
        my_end   = my_start + slice_size   (last rank may get fewer)

    gpSimFull is called with only the slice size, so memory scales with
    N_SIMS / nrank rather than N_SIMS.
    """
    # print(filename)
    ra, dec, band = radec_filename(filename)
    # print(band)
    # match = re.search(r'_z(\w)_', filename)
    # # print(file)
    # band  = match.group(1)

    # ── load LC (all ranks read independently — fast, file is small) ────────
    # time_arr, mag, mag_err, SNR = load_original_lc(ra, dec)
    time_arr, mag, mag_err = load_original_lc(ra, dec, filename)
    N       = len(time_arr)
    dur     = time_arr[-1]
    # N = int(round(dur)) + 1
    mag_med = np.median(mag)

    # ── DRW fit (all ranks do this independently — cheaper than broadcast) ──
    best_fit   = drw_fit(time_arr, mag, mag_err)
    SNR = best_fit[0] / np.median(mag_err)
    norm_amp = best_fit[0] / np.median(mag_err)
    # SNR = norm_amp if norm_amp >=1 else 1
    print(best_fit[0],SNR)

    DRW_kernel = DRW_term(*np.log(best_fit))
    print(f"[rank {rank}] RA={ra} DEC={dec} DRW fit done: {best_fit}", flush=True)

    # ── assign this rank's simulation slice ──────────────────────────────────
    base       = N_SIMS // nrank
    remainder  = N_SIMS  % nrank
    slice_size = base + (1 if rank < remainder else 0)

    if slice_size == 0:
        # more ranks than sims — this rank has nothing to do
        return time_arr, dur, [], []

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
        rows_daily   = []   # LC rows for SAVE_LC (one df per sim)
        rows_orgcad  = []
 
    # global sim index offset for this rank, so object_index is unique across ranks
    sim_offset   = rank * (N_SIMS // nrank) + min(rank, N_SIMS % nrank)

    # sim_times    = []
    # build cKDTree once — t_even is identical for all sims in this slice
    tree   = cKDTree(t_even.reshape(-1, 1))
    width  = len(str(slice_size))   # fixed width for timing print

    source_start = time.time()

    for i in range(slice_size):
        sim_start = time.time()
        object_idx = sim_offset + i

        y_i    = y_batch[i] + mag_med
        yerr_i = yerr_batch[i]

        sim, simcad = build_simcad(t_even, y_i, yerr_i, time_arr,tree)

        sf_full_dict = SF_wnoise(sim["mag"],    sim["time"],    sim["err"], color="#436BAD")
                                #  save_plt=True,path=OUTPUT_PATH+f'{ra}_{dec}/z{band}_sci_merged_{object_idx}_')
        sf_cad_dict  = SF_wnoise(simcad["mag"], simcad["time"], simcad["err"], color="orange")
                                #  save_plt=True,path=OUTPUT_PATH+f'{ra}_{dec}/z{band}_sci_merged_{object_idx}_cad')

        fit_full = SF_linmix(sf_full_dict[0])
        fit_cad  = SF_linmix(sf_cad_dict[0])

        # if fit_cad is not None and fit_full is not None:
        #     try:
        #         dailycad.append(fit_full["_gamma"])
        #         orgcad.append(fit_cad["_gamma"])
        #     except Exception as e:
        #         print(f"[rank {rank}]   sim {i} key error: {e}", flush=True)

        # ── determine fit validity and reason ────────────────────────────────
        valid      = False
        fail_reason = ""
        if fit_full is None:
            fail_reason = "SF_linmix returned None (daily)"
        elif fit_cad is None:
            fail_reason = "SF_linmix returned None (orgcad)"
        else:
            try:
                # dailycad.append(fit_full["_gamma"])
                # orgcad.append(fit_cad["_gamma"])
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

    return time_arr, dur, dailycad, orgcad


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── rank 0: build the work list ──────────────────────────────────────────
    job_start = time.time()
    if rank == 0:
        # if SAVE_LC:
        #     os.makedirs(LC_PATH, exist_ok=True)

        # all_files = sorted(glob.glob(DATA_PATH + "corrlc_*.csv"))
        all_files = sorted(glob.glob(DATA_PATH + "*sci_merged.parquet"))
        # ll_files = sorted(glob.glob(DATA_PATH + "*_z[gri]_merged.parquet"))
        done      = load_checkpoint()
        todo      = []
        skipped   = 0

        for f in all_files[:20]:
            try:
                print(f)
                ra, dec, band = radec_filename(f)
                print(band)
            except ValueError:
                continue
            key = (ra, dec)
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
        time_arr, dur, dailycad, orgcad = None, None, [], []

        try:
            time_arr, dur, dailycad, orgcad = process_slice(f)
        except Exception as e:
            print(f"[rank {rank}] ERROR on {f}: {e}", flush=True)
            dailycad, orgcad = [], []      # explicit: failed ranks still contribute empty

        # ── gather all slices to rank 0 (MPI-synchronised, no filesystem race) 
        all_slices = comm.gather((dailycad, orgcad), root=0)

        # ── rank 0 merges and saves immediately ──────────────────────────────
        if rank == 0:
            try:
                ra, dec, band = radec_filename(f)
                if time_arr is None:                       # this rank's own load failed
                    print(f"[rank 0] skip merge for {f} (no data)", flush=True)
                else:
                    merge_and_save(ra, dec, band, time_arr, dur, all_slices)
                # time_arr/dur come from rank 0's own process_slice call
                # merge_and_save(ra, dec, band, time_arr, dur, all_slices)
                print(f"[rank 0] source done in {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[rank 0] merge ERROR for {f}: {e}", flush=True)

    comm.Barrier()
    # MPI.Finalize()
    # sys.exit(0)

    if rank == 0:
        done = load_checkpoint()
        total = time.time() - job_start
        print(f"[rank 0] all done. {len(done)} sources in {OUTPUT_PKL}.", flush=True)
        print(f"[rank 0] total walltime: {total:04.1f}s", flush=True)



if __name__ == "__main__":
    main()