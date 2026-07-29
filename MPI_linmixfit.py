#!/usr/bin/env python3
"""
run_SFlinmix_mw.py
Master–worker MPI driver for SF_linmix over the per-source SF pkl files
produced by optSF (~/results/partials/{ra}_{dec}/{parquet_basename}_{x}cs.pkl).

Usage on the cluster:
    mpirun -n <nranks> python run_SFlinmix_mw.py
    nranks >= 2 : rank 0 = master (dispatches, does not compute),
                  ranks 1..n-1 = workers
    nranks == 1 : serial fallback (processes everything itself)

Pattern
───────
Dynamic load balancing: the master holds the file list and hands out one
file at a time. A worker fits it, reports back, and receives the next one.
No rank ever idles while work remains — unlike round-robin, where a rank
stuck with the slow fits leaves the others waiting at the barrier.

Safety (per the stage-3 lessons):
  * SIGALRM timeout (300 s) around each SF_linmix call — LinMix can hang
    non-deterministically
  * parallelize=False is already enforced inside SF_linmix
  * per-file try/except in workers — one bad fit never kills the job
  * resume support: inputs whose output pkl already exists are skipped
  * outputs are saved by THIS driver, named from the input file.
    SF_linmix's own save_pkl is NOT used: it names files by RA only,
    so the g/r/i bands of the same source would overwrite each other.
"""

import os
import glob
import time
import signal
import pickle
import datetime
import traceback
from mpi4py import MPI
from newSF import SF_linmix

# ── configuration ─────────────────────────────────────────────────────────────
X          = 10                                    # must match the optSF run
INPUT_GLOB = os.environ["HOME"] + f"/results/partials/*/*_{X}cs.pkl"
FIT_SUFFIX = "_linmixfit.pkl"                      # output: <input stem> + this
TIMEOUT_S  = 300
SAVE_PLOT  = True
MTIME_DAY  = "2026-07-23"                          # None = no date filter;
                                                   # else keep files modified on this day

LINMIX_KWARGS = dict(
    phot     = False,
    verbose  = False,
    amp_at   = 365,
    plot     = False,
    save_pkl = False,      # driver saves instead — see docstring
)

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
    """.../foo_10cs.pkl → .../foo_10cs_linmixfit.pkl"""
    return pkl_file[:-len(".pkl")] + FIT_SUFFIX


def build_file_list():
    all_files = sorted(
        f for f in glob.glob(INPUT_GLOB)
        if not f.endswith(FIT_SUFFIX)              # defensive: never refit outputs
    )

    # optional subset: keep only files modified on MTIME_DAY
    if MTIME_DAY is not None:
        target = datetime.date.fromisoformat(MTIME_DAY)
        all_files = [
            f for f in all_files
            if datetime.date.fromtimestamp(os.path.getmtime(f)) == target
        ]
        print(f"[master] date filter {MTIME_DAY}: {len(all_files)} files match",
              flush=True)

    todo = [f for f in all_files if not os.path.exists(out_name(f))]
    print(f"[master] {len(all_files)} SF pkl files found, "
          f"{len(all_files) - len(todo)} already fitted, {len(todo)} to do",
          flush=True)
    return todo


def process(pkl_file):
    """Load one SF dict, fit it with a hard timeout, save the result."""
    with open(pkl_file, "rb") as f:
        sf_dict = pickle.load(f)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)
    try:
        fitted = SF_linmix(sf_dict,
                           save_plot=SAVE_PLOT,
                           path=os.path.dirname(pkl_file) + "/",
                           **LINMIX_KWARGS)
    finally:
        signal.alarm(0)

    if fitted is None:
        return "skipped (SF_linmix returned None)"

    with open(out_name(pkl_file), "wb") as f:
        pickle.dump(fitted, f)
    return "done"


# ── master ────────────────────────────────────────────────────────────────────
def master(files):
    status   = MPI.Status()
    next_i   = 0
    n_active = nrank - 1

    # seed every worker with one file (or stop it immediately if none left)
    for w in range(1, nrank):
        if next_i < len(files):
            comm.send(files[next_i], dest=w, tag=TAG_WORK)
            next_i += 1
        else:
            comm.send(None, dest=w, tag=TAG_STOP)
            n_active -= 1

    # dispatch loop: each result frees a worker for the next file
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
        # serial fallback — no MPI messaging at all
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
