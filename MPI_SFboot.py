#!/usr/bin/env python3
"""
MPI_splcsfit.py
Master–worker MPI driver for starplaw.starplaw_from_parquet: per parent parquet,
select target + comparison stars (as optSF), build the SF on the target's epochs,
joint bootstrap, fit SF_obs = sqrt(PL^2 + (N SF_star)^2) in log10 SF.

Usage on the cluster:
    mpirun -n <nranks> python MPI_splcsfit.py [--force] [--method boot|gls]
                                              [--unit night|epoch] [--independent]
                                              [--nboot N] [--save-plot] [--store-samples]
                                              [--file P [P ...] | --list JSON] [--limit N]
    nranks >= 2 : rank 0 = master (dispatches, does not compute)
    nranks == 1 : serial fallback

Inputs (parent parquets, as runSF.py): default ~/BAT_results/*z[gri]_merged.parquet;
  --file   one or more parquets (bare names are looked up in ~/BAT_results)
  --list   JSON list of parquets, relative to ~/BAT_results unless absolute
  --limit  keep only the first N of those (before skipping already-fitted ones)
e.g.  python MPI_splcsfit.py --file 10.16620_+24.76092_zg_merged.parquet --save-plot

Output per parquet:  ~/results/partials/{ra}_{dec}/<parquet>_{X}cs_splcs[<tag>].pkl
  tag is empty for the default (boot, night, joint) and e.g. '_gls_night' or
  '_boot_night_indep' otherwise, so variants can be compared without overwriting.
--force refits even if that output already exists.
"""

import os
import io
import glob
import json
import time
import signal
import argparse
import contextlib
import traceback
from pathlib import Path
from mpi4py import MPI
import SF_boot as sp

# ── CLI (parsed on every rank; mpirun passes argv to all) ──────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--force", action="store_true", help="refit even if the output exists")
_ap.add_argument("--method", choices=["boot", "gls"], default="boot")
_ap.add_argument("--unit", choices=["night", "epoch"], default="night")
_ap.add_argument("--independent", action="store_true",
                 help="resample target and stars with independent draws (default: joint)")
_ap.add_argument("--nboot", type=int, default=200)
_ap.add_argument("--save-plot", action="store_true")
_ap.add_argument("--store-samples", action="store_true",
                 help="keep the posterior samples (samples_boot) in the output pkl")
_src = _ap.add_mutually_exclusive_group()
_src.add_argument("--file", nargs="+", help="parquet(s); bare names are looked up in ~/BAT_results")
_src.add_argument("--list", help="JSON list of parquets, relative to ~/BAT_results unless absolute")
_ap.add_argument("--limit", type=int, default=None, help="only the first N input files")
ARGS = _ap.parse_known_args()[0]
FORCE, METHOD, UNIT, JOINT = ARGS.force, ARGS.method, ARGS.unit, not ARGS.independent
TAG = "" if (METHOD, UNIT, JOINT) == ("boot", "night", True) else \
      f"_{METHOD}_{UNIT}" + ("" if JOINT else "_indep")

# ── configuration ─────────────────────────────────────────────────────────────
X            = 10                                   # comparison stars, as in runSF.py
SIGMA_FILTER = True                                 # as in runSF.py's OPTsf_KWARGS
DATA_DIR     = Path(os.environ["HOME"]) / "BAT_results"
INPUT_GLOB   = str(DATA_DIR / "*z[gri]_merged.parquet")   # as runSF.py
TIMEOUT_S    = 900                                  # selection + bootstrap + refits per source

TAG_WORK, TAG_RESULT, TAG_STOP = 1, 2, 3
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ── helpers ───────────────────────────────────────────────────────────────────
class FitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise FitTimeout


def _resolve(p):
    """absolute or existing path as given; otherwise relative to DATA_DIR."""
    p = Path(os.path.expanduser(p))
    return str(p if p.is_absolute() or p.exists() else DATA_DIR / p)


def build_file_list():
    if ARGS.file:
        files = [_resolve(p) for p in ARGS.file]
    elif ARGS.list:
        files = [_resolve(p) for p in json.loads(Path(os.path.expanduser(ARGS.list)).read_text())]
    else:
        files = sorted(glob.glob(INPUT_GLOB))
    files = files[:ARGS.limit]
    missing = [f for f in files if not os.path.exists(f)]
    for f in missing:
        print(f"[master] not found, skipped: {f}", flush=True)
    files = [f for f in files if f not in missing]
    todo = files if FORCE else [f for f in files
                                if not os.path.exists(sp.output_path(f, x=X, tag=TAG))]
    print(f"[master] {len(files)} parquets, {len(files) - len(todo)} already fitted, "
          f"{len(todo)} to do | method={METHOD} unit={UNIT} joint={JOINT} nboot={ARGS.nboot}"
          + ("  [--force]" if FORCE else ""), flush=True)
    return todo


def process(file):
    """Fit one parquet under a hard timeout; the noisy selection printouts are
    captured and only the last line is reported when the source is skipped."""
    buf = io.StringIO()
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)
    try:
        with contextlib.redirect_stdout(buf):
            res = sp.starplaw_from_parquet(file, x=X, sigma_filter=SIGMA_FILTER,
                                           n_boot=ARGS.nboot, unit=UNIT, joint=JOINT,
                                           method=METHOD, save=True,
                                           save_plot=ARGS.save_plot, tag=TAG,
                                           store_samples=ARGS.store_samples)
    finally:
        signal.alarm(0)
    if res is None:
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        return "skipped (" + (lines[-1] if lines else "no reason printed") + ")"
    return (f"done gamma={res['gamma_boot']:.3f} A_365={res.get('A_365_boot', float('nan')):.3f} "
            f"N={res['N_boot']:.2f} | {res['flag_boot']}")


# ── master ────────────────────────────────────────────────────────────────────
def master(files):
    status   = MPI.Status()
    next_i   = 0
    n_active = nrank - 1
    for w in range(1, nrank):
        if next_i < len(files):
            comm.send(files[next_i], dest=w, tag=TAG_WORK); next_i += 1
        else:
            comm.send(None, dest=w, tag=TAG_STOP); n_active -= 1
    n_done = 0
    while n_active > 0:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        n_done += 1
        print(f"[master] ({n_done}/{len(files)}) {msg}", flush=True)
        w = status.Get_source()
        if next_i < len(files):
            comm.send(files[next_i], dest=w, tag=TAG_WORK); next_i += 1
        else:
            comm.send(None, dest=w, tag=TAG_STOP); n_active -= 1


# ── worker ────────────────────────────────────────────────────────────────────
def _run(f):
    t0 = time.time()
    try:
        res = process(f)
    except FitTimeout:
        res = f"TIMEOUT after {TIMEOUT_S}s"
    except Exception:
        res = f"ERROR:\n{traceback.format_exc()}"
    return f"{res} in {time.time() - t0:.1f}s — {os.path.basename(f)}"


def worker():
    status = MPI.Status()
    while True:
        f = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            return
        comm.send(f"[rank {rank}] " + _run(f), dest=0, tag=TAG_RESULT)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if nrank == 1:
        files = build_file_list()
        for n, f in enumerate(files, 1):
            print(f"({n}/{len(files)}) " + _run(f), flush=True)
        return
    if rank == 0:
        t0 = time.time()
        master(build_file_list())
        h, rem = divmod(time.time() - t0, 3600); m, s = divmod(rem, 60)
        print(f"[master] all done. walltime: {int(h):02d}h {int(m):02d}m {s:04.1f}s", flush=True)
    else:
        worker()


if __name__ == "__main__":
    main()