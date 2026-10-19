#!/usr/bin/env python3
"""
MPI_splcsfit.py
Master–worker MPI driver for SF_starplaw (observed SF = star floor ⊕ power law,
NO subtraction) over the per-source SF pkls produced by optSF.
 
Usage on the cluster:
    mpirun -n <nranks> python MPI_splcsfit.py [--force]
    nranks >= 2 : rank 0 = master (dispatches, does not compute)
    nranks == 1 : serial fallback
 
--force :
  * refit even if the _splcsfit.pkl output already exists, AND
  * if an input _{X}cs.pkl predates the SF_obs/SF_cs patch (no forward-model
    keys), re-run optSF on its parent parquet to regenerate+overwrite the input,
    then fit. Without --force such inputs are skipped (not silently recomputed).
"""
 
import os
import glob
import json
import time
import signal
import pickle
import argparse
import datetime
import traceback
from pathlib import Path
from mpi4py import MPI
from newSF import SF_starplaw, optSF
 
# ── CLI ─────────────────────────────────────────────────────────────────────────
# parsed on every rank (mpirun passes argv to all) → no bcast needed
_ap = argparse.ArgumentParser()
_ap.add_argument("--force", action="store_true",
                 help="refit existing outputs; regenerate inputs lacking SF_obs/SF_cs via optSF")
FORCE = _ap.parse_known_args()[0].force
 
# ── configuration ─────────────────────────────────────────────────────────────
X          = 10                                    # must match the optSF run
# INPUT_GLOB = os.environ["HOME"] + f"/results/partials/*/*_{X}cs.pkl"
INPUT_GLOB = os.environ["HOME"] + f"/results/partials/10.1662_24.76092/10.16620_+24.76092_zg_merged.parquet_10cs.pkl"
# INPUT_GLOB = os.environ["HOME"] + f"/results/partials/0.61008_3.35193/0.61008_+3.35193_zg_merged.parquet_10cs.pkl"
FIT_SUFFIX = "_2cfit.pkl"                          # output: <input stem> + this
 # output: <input stem> + this
TIMEOUT_S  = 300
SAVE_PLOT  = True
MTIME_DAY  = None                                  # None, or "YYYY-MM-DD" cutoff
N_SOURCES  = None                                  # None = all; else cap to first N
 
# restrict inputs to the _{X}cs.pkl produced by runSF.py from this parent list.
DATA_DIR  = Path(os.environ["HOME"]) / "BAT_results"   # where the parent parquets live
FILE_LIST = None
# FILE_LIST = DATA_DIR / "tosync.json"
 
# optSF kwargs for regenerating a stale input — MUST match runSF.py's OPTsf_KWARGS
OPTSF_KWARGS = dict(save=True, sigma_filter=True, x=X, save_plt=SAVE_PLOT)
 
STARPLAW_KWARGS = dict(amp_at=365, free_N=True, verbose=False)
 
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
    """.../foo_10cs.pkl → .../foo_10cs_splcsfit.pkl"""
    return pkl_file[:-len(".pkl")] + FIT_SUFFIX
 
 
def allowed_pkl_names():
    parents = json.loads(Path(FILE_LIST).read_text())
    return {f"{os.path.basename(p)}_{X}cs.pkl" for p in parents}
 
 
def _unwrap(d):
    """run_txt_sf wraps as {'SF_dict':[d]}; optSF writes the dict directly."""
    if isinstance(d, dict) and "SF_dict" in d:
        inner = d["SF_dict"][0]
        inner.setdefault("RA", d.get("object"))
        return inner
    if isinstance(d, list):
        return d[0]
    return d
 
 
def _has_patch(sf_dict):
    return (isinstance(sf_dict, dict)
            and sf_dict.get("SF_obs") is not None
            and sf_dict.get("SF_cs") is not None)
 
 
def _parent_parquet(pkl_file):
    """.../<parquet_basename>_{X}cs.pkl → <DATA_DIR>/<parquet_basename>"""
    base = os.path.basename(pkl_file)
    stem = base[:-len(f"_{X}cs.pkl")]              # keeps the trailing '.parquet'
    return str(DATA_DIR / stem)
 
 
def build_file_list():
    all_files = sorted(
        f for f in glob.glob(INPUT_GLOB)
        if not f.endswith(FIT_SUFFIX)              # never refit our own outputs
        and "_linmixfit" not in f                  # nor the other fitters' outputs
        and "bplfit" not in f
        and "splcsfit" not in f
    )
 
    if FILE_LIST is not None:
        allowed = allowed_pkl_names()
        all_files = [f for f in all_files if os.path.basename(f) in allowed]
        print(f"[master] file-list {os.path.basename(str(FILE_LIST))}: "
              f"{len(all_files)}/{len(allowed)} of the listed parents present", flush=True)
 
    all_files = all_files[:N_SOURCES]
 
    if MTIME_DAY is not None:
        cutoff = datetime.datetime.fromisoformat(MTIME_DAY).timestamp()
        all_files = [f for f in all_files if os.path.getmtime(f) >= cutoff]
        print(f"[master] mtime >= {MTIME_DAY}: {len(all_files)} files match", flush=True)
 
    todo = all_files if FORCE else [f for f in all_files if not os.path.exists(out_name(f))]
    print(f"[master] {len(all_files)} SF pkl files found, "
          f"{len(all_files) - len(todo)} already fitted, {len(todo)} to do"
          + ("  [--force]" if FORCE else ""), flush=True)
    return todo
 
 
def process(pkl_file):
    """Load one SF dict (regenerating it via optSF if stale and --force), fit it
    under a hard timeout, save the result."""
    with open(pkl_file, "rb") as f:
        sf_dict = _unwrap(pickle.load(f))
 
    note = ""
    if not _has_patch(sf_dict):
        if not FORCE:
            return "skipped (no SF_obs/SF_cs — rerun optSF or use --force)"
        parquet = _parent_parquet(pkl_file)
        if not os.path.exists(parquet):
            return f"skipped (--force regen: parent parquet not found {os.path.basename(parquet)})"
        # regen OUTSIDE the alarm: optSF is heavier and shouldn't trip the fit timeout
        sf_dict = optSF(parquet, **OPTSF_KWARGS)      # recomputes + overwrites the _{X}cs.pkl
        if sf_dict is None or not _has_patch(sf_dict):
            return "skipped (optSF regen returned None / still no SF_obs/SF_cs)"
        note = " (regenerated input)"
 
    plot_path = out_name(pkl_file)[:-len(".pkl")] + ".png" if SAVE_PLOT else None
 
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_S)
    try:
        fitted = SF_starplaw(sf_dict, save_plot=SAVE_PLOT, path=plot_path, **STARPLAW_KWARGS)
    finally:
        signal.alarm(0)
 
    if fitted is None:
        return "skipped (SF_starplaw returned None)" + note
 
    with open(out_name(pkl_file), "wb") as f:
        pickle.dump(fitted, f)
    return "done" + note
 
 
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
            print(f"({i}/{len(files)}) {res} in {time.time() - t0:.1f}s — {f}", flush=True)
        return
 
    if rank == 0:
        job_start = time.time()
        files = build_file_list()
        master(files)
        total = time.time() - job_start
        hours, rem = divmod(total, 3600)
        mins, secs = divmod(rem, 60)
        print(f"[master] all done. "
              f"walltime: {int(hours):02d}h {int(mins):02d}m {secs:04.1f}s", flush=True)
    else:
        worker()
 
 
if __name__ == "__main__":
    main()
 