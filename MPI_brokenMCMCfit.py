#!/usr/bin/env python3
"""
run_bpl_mw.py
Master–worker MPI driver for bpl_mcmc (broken-power-law emcee fit) over the
per-source SF pkls produced by optSF.

Usage on the cluster:
    mpirun -n <nranks> python run_bpl_mw.py
    nranks >= 2 : rank 0 = master (dispatches, does not compute),
                  ranks 1..n-1 = workers
    nranks == 1 : serial fallback

Same structure as run_SFlinmix_mw.py; differences from that driver:
  * fits with bpl_mcmc instead of SF_linmix (different model + output schema)
  * output suffix is _bplfit.pkl, so the two fitters never collide and each
    keeps independent resume state
  * bpl_mcmc doesn't save or plot — the driver saves; all plotting stays off
  * progress=False so emcee's tqdm bars don't flood the job log
"""

import os
import glob
import time
import signal
import pickle
import datetime
import traceback
from mpi4py import MPI
from newSF import bpl_mcmc

# ── configuration ─────────────────────────────────────────────────────────────
X          = 10                                    # must match the optSF run
# MODEL      = "flat"                                 # "flat" (A at dt=1) or "at_break" (A at break)
MODEL      = "at_break"                                 # "flat" (A at dt=1) or "at_break" (A at break)
INPUT_GLOB = os.environ["HOME"] + f"/results/partials/*/*_{X}cs.pkl"
FIT_SUFFIX = f"_bpl_{MODEL}fit.pkl"                 # output: <input stem> + this (model in name)
TIMEOUT_S  = 300
# MTIME_DAY  = None                                   # None = no date filter;
MTIME_DAY  = "2026-08-19"                                   # None = no date filter;
                                                    # else "YYYY-MM-DD" to subset
N_SOURCES = None
SAVE_PLOT  = False                                  # save per-source bpl fit PNGs
VERBOSE    = False                                  # per-fit prints (spams the log)

TAG_WORK, TAG_RESULT, TAG_STOP = 1, 2, 3

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ── helpers ───────────────────────────────────────────────────────────────────
class FitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise FitTimeout


def out_name(pkl_file):
    """.../foo_10cs.pkl → .../foo_10cs_bplfit.pkl"""
    return pkl_file[:-len(".pkl")] + FIT_SUFFIX


def build_file_list():
    all_files = sorted(
        f for f in glob.glob(INPUT_GLOB)
        if not f.endswith(FIT_SUFFIX)              # never refit our own outputs
        and "_linmixfit" not in f                  # nor the linmix driver's outputs
    )[:N_SOURCES]

    if MTIME_DAY is not None:
        cutoff = datetime.datetime.fromisoformat(MTIME_DAY).timestamp()
        all_files = [f for f in all_files if os.path.getmtime(f) >= cutoff]
        print(f"[master] mtime >= {MTIME_DAY}: {len(all_files)} files match", flush=True)

    todo = [f for f in all_files if not os.path.exists(out_name(f))]
    print(f"[master] {len(all_files)} SF pkl files found, "
          f"{len(all_files) - len(todo)} already fitted, {len(todo)} to do",
          flush=True)
    return todo


def process(pkl_file):
    """Load one SF dict, fit it with a hard timeout, save the result."""
    with open(pkl_file, "rb") as f:
        sf_dict = pickle.load(f)

    # per-source PNG next to the output pkl (unique per input, avoids RA/band collisions)
    plot_path = out_name(pkl_file)[:-len(".pkl")] + ".png" if SAVE_PLOT else None

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)
    try:
        fitted = bpl_mcmc(sf_dict, plot=False, model_check=False, mcmc_check=False,
                          progress=False, save_plot=SAVE_PLOT, path=plot_path,
                          verbose=VERBOSE, model=MODEL)
    finally:
        signal.alarm(0)

    if fitted is None:
        return "skipped (bpl_mcmc returned None)"

    with open(out_name(pkl_file), "wb") as f:
        pickle.dump(fitted, f)
    return "done"


# ── master ────────────────────────────────────────────────────────────────────
def master(files):
    status   = MPI.Status()
    next_i   = 0
    n_active = nrank - 1

    for w in range(1, nrank):
        if next_i < len(files):
            comm.send(files[next_i], dest=w, tag=TAG_WORK)
            next_i += 1
        else:
            comm.send(None, dest=w, tag=TAG_STOP)
            n_active -= 1

    n_done = 0
    while n_active > 0:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        n_done += 1
        print(f"[master] ({n_done}/{len(files)}) {msg}", flush=True)

        w = status.Get_source()
        if next_i < len(files):
            comm.send(files[next_i], dest=w, tag=TAG_WORK)
            next_i += 1
        else:
            comm.send(None, dest=w, tag=TAG_STOP)
            n_active -= 1


# ── worker ────────────────────────────────────────────────────────────────────
def worker():
    status = MPI.Status()
    while True:
        f = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            return

        t0 = time.time()
        try:
            res = process(f)
        except FitTimeout:
            res = f"TIMEOUT after {TIMEOUT_S}s"
        except Exception:
            res = f"ERROR:\n{traceback.format_exc()}"

        comm.send(f"[rank {rank}] {res} in {time.time() - t0:.1f}s — {f}",
                  dest=0, tag=TAG_RESULT)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if nrank == 1:
        files = build_file_list()
        for i, f in enumerate(files, 1):
            t0 = time.time()
            try:
                res = process(f)
            except FitTimeout:
                res = f"TIMEOUT after {TIMEOUT_S}s"
            except Exception:
                res = f"ERROR:\n{traceback.format_exc()}"
            print(f"({i}/{len(files)}) {res} in {time.time() - t0:.1f}s — {f}",
                  flush=True)
        return

    if rank == 0:
        job_start = time.time()
        files = build_file_list()
        master(files)
        total = time.time() - job_start
        hours, rem = divmod(total, 3600)
        mins, secs = divmod(rem, 60)
        print(f"[master] all done. "
              f"walltime: {int(hours):02d}h {int(mins):02d}m {secs:04.1f}s",
              flush=True)
    else:
        worker()


if __name__ == "__main__":
    main()