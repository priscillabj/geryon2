"""
run_simulations.py

Usage on the cluster:
    mpirun -n <nranks> python run_simulations.py

Requirements:
    mpi4py, eztao, numpy, pandas, scipy, linmix, celerite
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from functools import partial
from scipy.spatial import cKDTree
from mpi4py import MPI

from eztao.ts import drw_fit
from eztao.carma import DRW_term
from eztao.ts import gpSimFull
from celerite import GP

# ── import your SF functions ────────────────────────────────────────────────
# Adjust this import to wherever SF_wnoise and SF_linmix live in your project
from SF import SF_wnoise, SF_linmix

# ── configuration ────────────────────────────────────────────────────────────
N_SIMS      = 10_000   # simulations per source
OUTPUT_PKL  = "simfit_results.pkl"
# DATA_PATH   = os.environ["HOME"] + "/Nextcloud/Doutorado/Forced_Phot/"
DATA_PATH   = os.environ["HOME"] + "/LC/"

# ── MPI setup ────────────────────────────────────────────────────────────────
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def radec_filename(filename):
    match = re.search(r"lc_(\d+\.\d+)_(-?\d+\.\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Cannot parse RA/DEC from filename: {filename}")


def load_original_lc(ra, dec):
    """Load the original ZTF g-band light curve for one source."""
    #path = DATA_PATH + f"calibration_sources/PS1/{ra}_{dec}/caltarget_flux_tot_clr_uJy.parquet"
    path = DATA_PATH + f"corrlc_{ra}_{dec}.csv"
    # ztf  = pd.read_parquet(path)
    ztf = pd.read_csv(path)
    ztf  = ztf[ztf["filter"] == "ZTF_g"]

    time_arr = ztf.mjd.values.copy()
    time_arr -= time_arr[0]
    mag      = ztf.magtot_clr_corr.values
    mag_err  = ztf.magunc_clr_corr.values

    flux     = ztf.flux_tot_clr_uJy_corr.values
    flux_err = ztf.fluxunc_tot_clr_uJy_corr_alt2.values
    snr_med  = np.nanmedian(flux / flux_err)
    SNR      = 20 if np.isnan(snr_med) else min(snr_med, 20)

    return time_arr, mag, mag_err, SNR


def build_simcad(t_even, y_batch, yerr_batch, time_orig):
    """
    Match one simulated (even-cadence) LC to the original cadence via nearest
    neighbour, exactly as in the original SimCadLC.

    t_even    : 1-D array, shape (N,)
    y_batch   : 1-D array, shape (N,)  — one LC
    yerr_batch: 1-D array, shape (N,)
    time_orig : 1-D array — original MJD stamps (shifted to start at 0)

    Returns sim DataFrame (even cadence) and simcad DataFrame (original cadence).
    """
    mag_median = 0.0   # median is added by the caller

    sim = pd.DataFrame({"time": t_even, "mag": y_batch, "err": yerr_batch})

    tree    = cKDTree(t_even.reshape(-1, 1))
    _, idx  = tree.query(time_orig.reshape(-1, 1))
    simcad  = pd.DataFrame({
        "time": t_even[idx],
        "mag" : y_batch[idx],
        "err" : yerr_batch[idx],
    })
    return sim, simcad


def process_one_source(filename, n_sims=N_SIMS):
    """
    Full pipeline for a single source.  Called by every worker rank.

    Returns a dict ready to append to simfit_results.
    """
    ra, dec = radec_filename(filename)
    print(f"[rank {rank}] processing RA={ra} DEC={dec}", flush=True)

    # ── 1. load original LC ──────────────────────────────────────────────────
    time_arr, mag, mag_err, SNR = load_original_lc(ra, dec)
    N      = len(time_arr)
    dur    = time_arr[-1]
    mag_med = np.median(mag)

    # ── 2. fit DRW  (once per source, not per simulation) ───────────────────
    best_fit   = drw_fit(time_arr, mag, mag_err)
    DRW_kernel = DRW_term(*np.log(best_fit))
    print(f"[rank {rank}]   DRW fit done: {best_fit}", flush=True)

    # ── 3. generate ALL simulated LCs in one batched call ───────────────────
    #    gpSimFull with nLC=n_sims returns arrays of shape (n_sims, N)
    t_batch, y_batch, yerr_batch = gpSimFull(
        DRW_kernel, SNR, dur, N, nLC=n_sims, log_flux=False
    )
    # t_batch is (n_sims, N) but all rows are identical (uniform grid)
    t_even = t_batch[0]

    # ── 4. loop over simulations ─────────────────────────────────────────────
    dailycad  = []
    orgcad    = []

    for i in range(n_sims):
        print('\n')
        print(f'simulation #{i+1}')
        y_i    = y_batch[i]    + mag_med
        yerr_i = yerr_batch[i]

        sim, simcad = build_simcad(t_even, y_i, yerr_i, time_arr)

        # SF for daily-cadence sim (full even grid)
        sf_full_dict = SF_wnoise(
            sim["mag"], sim["time"], sim["err"],
            plot=False, color="#436BAD"
        )
        # SF for original-cadence sim
        sf_cad_dict  = SF_wnoise(
            simcad["mag"], simcad["time"], simcad["err"],
            plot=False, color="orange"
        )

        fit_full = SF_linmix(sf_full_dict[0])
        fit_cad  = SF_linmix(sf_cad_dict[0])

        if fit_cad is not None and fit_full is not None:
            try:
                dailycad.append(fit_full["_gamma"])
                orgcad.append(fit_cad["_gamma"])
            except Exception as e:
                print(f"[rank {rank}]   sim {i} key error: {e}", flush=True)

        if (i + 1) % 500 == 0:
            print(f"[rank {rank}]   {i+1}/{n_sims} sims done", flush=True)

    result = {
        "RA"         : ra,
        "DEC"        : dec,
        "ref_band"   : "g",
        "n_epochs"   : N,
        "length"     : dur,
        "mean_cad"   : float(np.mean(np.diff(time_arr))),
        "n_sims"     : len(dailycad),          # actual completed sims
        "1day_gamma" : dailycad,
        "orgcad_gamma": orgcad,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint():
    """Return existing results dict keyed by (ra, dec), or empty dict."""
    if os.path.exists(OUTPUT_PKL):
        with open(OUTPUT_PKL, "rb") as f:
            existing = pickle.load(f)
        return {(r["RA"], r["DEC"]): r for r in existing}
    return {}


def save_checkpoint(results_dict):
    """Overwrite the pickle with current results."""
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(list(results_dict.values()), f)


def is_complete(result, n_sims=N_SIMS):
    """A result is complete only if it has the expected number of sims."""
    return result.get("n_sims", 0) >= n_sims


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── rank 0: build the work list ──────────────────────────────────────────
    if rank == 0:
        # collect filenames (adjust glob/path as needed)
        import glob
        all_files = sorted(glob.glob(DATA_PATH + "corrlc_*.csv"))

        # load checkpoint and filter out already-complete sources
        done     = load_checkpoint()
        todo     = []
        skipped  = 0
        for f in all_files[:5]:
            try:
                ra, dec = radec_filename(f)
            except ValueError:
                continue
            key = (ra, dec)
            if key in done and is_complete(done[key]):
                skipped += 1
            else:
                todo.append(f)

        print(f"[rank 0] {len(all_files)} total sources, "
              f"{skipped} already complete, {len(todo)} to process.", flush=True)
    else:
        todo = None
        done = None

    # ── broadcast todo list to all ranks ────────────────────────────────────
    todo = comm.bcast(todo, root=0)

    # ── distribute work: each rank takes every nrank-th file ─────────────────
    my_files = todo[rank::nrank]
    print(f"[rank {rank}] assigned {len(my_files)} sources", flush=True)

    # ── each rank processes its sources sequentially ─────────────────────────
    my_results = []
    for f in my_files:
        t0 = time.time()
        try:
            result = process_one_source(f)
            my_results.append(result)
            print(f"[rank {rank}] done in {time.time()-t0:.1f}s — "
                  f"{result['n_sims']} valid sims", flush=True)
        except Exception as e:
            print(f"[rank {rank}] ERROR on {f}: {e}", flush=True)

    # ── gather all results to rank 0 ─────────────────────────────────────────
    # gather returns a list of lists (one per rank)
    all_results = comm.gather(my_results, root=0)

    # ── rank 0: merge and save ───────────────────────────────────────────────
    if rank == 0:
        # reload checkpoint (may have been updated by a previous partial run)
        done = load_checkpoint()

        for rank_results in all_results:
            for result in rank_results:
                key = (result["RA"], result["DEC"])
                done[key] = result
                save_checkpoint(done)
                print(f"[rank 0] saved result for RA={result['RA']} "
                      f"DEC={result['DEC']}", flush=True)

        print(f"[rank 0] all done. {len(done)} sources in {OUTPUT_PKL}.", flush=True)


if __name__ == "__main__":
    main()
