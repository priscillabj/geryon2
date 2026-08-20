#!/usr/bin/env python
"""SF_wnoise over per-object RA_DEC txt light-curve trees. MPI, resumable.

Tree layout per object (RUN = ztfphot_stars_*_field*_ccd*_qid*):

    <root>/<RA_DEC>/<RUN>/psd_out/mhps_input/                 <- 1 target LC
    <root>/<RA_DEC>/<RUN>/psd_out/mhps_input_cs_lightcurves/  <- 1 txt per cal star
    <root>/<RA_DEC>/<RUN>/sf_out/<RA_DEC>.pkl                 <- written here
    <root>/<RA_DEC>/<RUN>/sf_out/plots/<RA_DEC>_sf.png        <- written here

Requires the patched SF_wnoise with an `out_name` parameter, where
    savefig(path + f"{out_name}.png")
i.e. `path` is a PREFIX (not a directory) and `out_name` is a stem.

Parallelism: round-robin across objects, no MPI collectives in the work loop.
Resume: per-object pickle in <RUN>/sf_out; existing pickle == done.
Only the per-rank JSONL logs go to --out.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import inspect
import json
import pickle
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

sys.path.insert(0, os.path.expanduser("~/git"))
import newSF
from newSF import SF_wnoise

ZP_UJY = 23.9          # m_AB = 23.9 - 2.5 log10(f / uJy)
COL = ["OBSMJD", "flux", "flux_err"]
OUT_NAME = "sf"                       # SF_wnoise does savefig(path + f"{out_name}.png")
LEGACY_PNG = "simulated_SF.png"       # pre-patch filename; only used by --restart
RA_DEC = re.compile(r"^\d+\.\d+_[+-]?\d+\.\d+$")


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("SF_wnoise exceeded timeout")


def read_lc(p, max_magerr=None):
    """Read MJD/flux/flux_err. max_magerr is applied as the equivalent SNR cut:
    sigma_mag = (2.5/ln10) * sigma_f/f < max_magerr  <=>  f/sigma_f > 1.0857/max_magerr
    """
    df = pd.read_csv(p, sep=r"\s+", header=None, comment="#", names=COL)
    df = df[(df["flux"] > 0) & (df["flux_err"] > 0) & np.isfinite(df["flux_err"])]
    if max_magerr:
        snr_min = (2.5 / np.log(10)) / max_magerr
        df = df[df["flux"] / df["flux_err"] >= snr_min]
    return df


def find_run(objdir):
    """Return the single ztfphot_stars_* run dir, or raise with the full list."""
    runs = sorted(p for p in objdir.glob("ztfphot_stars_*") if p.is_dir())
    if len(runs) != 1:
        raise ValueError(f"{len(runs)} run dirs: {[r.name for r in runs]}")
    return runs[0]


def load_source(run, use_cs=True, max_magerr=None, min_cs_epochs=10):
    psd = run / "psd_out"

    tgt = sorted((psd / "mhps_input").glob("*.txt"))
    if len(tgt) != 1:
        raise ValueError(f"{len(tgt)} target LCs: {[f.name for f in tgt]}")
    raw = read_lc(tgt[0])
    n_raw = len(raw)
    ztf = read_lc(tgt[0], max_magerr)

    cs_all = None
    cs_files = []
    if use_cs:
        cs_files = sorted((psd / "mhps_input_cs_lightcurves").glob("*.txt"))
        frames = [read_lc(f, max_magerr).assign(object_index=i, cs_file=f.name)
                  for i, f in enumerate(cs_files)]
        # a star gutted by the cut is worse than no star: it contributes a
        # sparse, badly-sampled SF to the noise-floor average
        frames = [d for d in frames if len(d) >= min_cs_epochs]
        if frames:
            cs_all = pd.concat(frames, ignore_index=True)

    return ztf, cs_all, len(cs_files), n_raw


def to_mag(df):
    mag = ZP_UJY - 2.5 * np.log10(df["flux"].values)
    err = (2.5 / np.log(10)) * df["flux_err"].values / df["flux"].values
    return mag, err, df["OBSMJD"].values


def sf_one(objdir, use_cs, min_epochs, timeout, make_plots, max_magerr=None):
    name = objdir.name
    rec = {"object": name, "run": None, "n_epochs": 0, "n_epochs_raw": 0, "n_cut": 0,
           "n_cs_files": 0, "n_cs_kept": 0, "n_cs_epochs": 0, "n_bins": 0,
           "valid": False, "fail_reason": None, "pkl": None, "png": None,
           "seconds": None}
    t0 = time.perf_counter()
    try:
        run = find_run(objdir)
        rec["run"] = run.name
        ztf, cs_all, n_cs, n_raw = load_source(run, use_cs, max_magerr)
        rec.update(n_epochs=len(ztf), n_epochs_raw=n_raw, n_cut=n_raw - len(ztf),
                   n_cs_files=n_cs,
                   n_cs_kept=0 if cs_all is None else cs_all["object_index"].nunique(),
                   n_cs_epochs=0 if cs_all is None else len(cs_all))
        if len(ztf) < min_epochs:
            raise ValueError(f"only {len(ztf)} target epochs (< {min_epochs})")

        sf_dir = run / "sf_out"
        sf_dir.mkdir(exist_ok=True)
        plot_dir = sf_dir / "plots"
        if make_plots:
            plot_dir.mkdir(parents=True, exist_ok=True)

        mag, merr, mjd = to_mag(ztf)
        png = plot_dir / f"{name}_{OUT_NAME}.png"

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            # patched SF_wnoise does savefig(path + f"{out_name}.png"):
            #   path     -> PREFIX (not a directory); trailing '_' is part of the name
            #   out_name -> stem only, no extension
            # save=False: the pickle branch writes path + f"{out_name}.pkl", i.e. into
            # plots/ and without os.replace -- a walltime kill mid-write would leave a
            # truncated file. This script does its own atomic write into sf_out/.
            SF_dict = SF_wnoise(
                mag, mjd, merr, cs_all=cs_all,logbin_max=np.log10(3000), verbose=False,
                save_plt=make_plots, save=False,
                path=f"{plot_dir}/{name}_" if make_plots else None,
                out_name=OUT_NAME)
        finally:
            signal.alarm(0)
            if make_plots:
                import matplotlib.pyplot as plt
                plt.close("all")   # SF_wnoise skips its close() if savefig raised

        if make_plots:
            rec["png"] = str(png) if png.exists() else None
            if rec["png"] is None:
                rec["warn"] = "save_plt=True but no png written"

        sf = SF_dict[0]["SF"].dropna()
        rec["n_bins"] = int(len(sf))
        if rec["n_bins"] == 0:
            raise ValueError("SF empty after dropna")

        payload = {"object": name, "run": run.name, "SF_dict": SF_dict,
                   "n_epochs": len(ztf), "n_cs_files": n_cs,
                   "cs_used": cs_all is not None, "zeropoint": ZP_UJY,
                   "max_magerr": max_magerr,
                   "mjd_range": [float(mjd.min()), float(mjd.max())]}
        out = sf_dir / f"{name}.pkl"
        tmp = sf_dir / f".{name}.pkl.tmp"          # same fs -> os.replace is atomic
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, out)
        rec["pkl"] = str(out)
        rec["valid"] = True

    except Exception as e:
        rec["fail_reason"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=-5)

    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def done_pkl(objdir):
    """Path of the resume sentinel, or None if the run dir is ambiguous."""
    try:
        return find_run(objdir) / "sf_out" / f"{objdir.name}.pkl"
    except Exception:
        return None


def survey(objdirs):
    for d in objdirs:
        runs = sorted(p.name for p in d.glob("ztfphot_stars_*") if p.is_dir())
        line = {"object": d.name, "n_runs": len(runs), "runs": runs}
        if len(runs) == 1:
            psd = d / runs[0] / "psd_out"
            line["n_target"] = len(list((psd / "mhps_input").glob("*.txt")))
            line["n_cs"] = len(list((psd / "mhps_input_cs_lightcurves").glob("*.txt")))
            line["writable"] = os.access(d / runs[0], os.W_OK)
        print(json.dumps(line), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing RA_DEC subdirs")
    ap.add_argument("--out", default=os.path.expanduser("~/results/txt_sf"),
                    help="per-rank JSONL logs only; SF and plots go in the tree")
    ap.add_argument("--no-cs", action="store_true", help="skip calibration stars")
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--max-magerr", type=float, default=0.5,
                    help="drop epochs with sigma_mag above this (0 disables); "
                         "applied to target and calibration stars alike")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per object")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N objects in sorted order (smoke test)")
    ap.add_argument("--plots", action="store_true",
                    help="save <RUN>/sf_out/plots/<RA_DEC>_sf.png per object")
    ap.add_argument("--restart", action="store_true",
                    help="DELETE <RUN>/sf_out/*.pkl and <RUN>/sf_out/plots/*_sf.png in "
                         "the selected objects, and truncate the JSONL logs")
    ap.add_argument("--survey", action="store_true",
                    help="rank 0 only: report multiplicity + writability, then exit")
    a = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    root = Path(os.path.expanduser(a.root))
    logdir = Path(os.path.expanduser(a.out)) / "logs"

    # every rank enumerates independently -> no bcast, no deadlock risk
    objdirs = sorted(p for p in root.iterdir() if p.is_dir() and RA_DEC.match(p.name))
    n_total = len(objdirs)
    # sorted order is identical on every rank, so this slice needs no communication
    if a.limit is not None:
        objdirs = objdirs[:a.limit]

    if a.survey:
        if rank == 0:
            survey(objdirs)
        return

    use_cs = not a.no_cs

    if rank == 0:
        logdir.mkdir(parents=True, exist_ok=True)
        if a.plots and "out_name" not in inspect.signature(newSF.SF_wnoise).parameters:
            print("ABORT: --plots needs the out_name parameter in SF_wnoise; without "
                  "it every object writes the same hardcoded filename.", flush=True)
            comm.Abort(1)
        if use_cs:
            src = inspect.getsource(newSF.SF_wnoise)
            if "'CCDquadID' in ztf.columns" not in src and \
               '"CCDquadID" in ztf.columns' not in src:
                print("ABORT: SF_wnoise still filters on ztf['CCDquadID'], which the "
                      "array-built target frame never has. Apply the guard patch or "
                      "run with --no-cs.", flush=True)
                comm.Abort(1)
        if a.restart:
            n = 0
            for d in objdirs:
                try:
                    run = find_run(d)
                except Exception:
                    continue
                sf_dir = run / "sf_out"
                for f in list(sf_dir.glob("*.pkl")) + \
                         list(sf_dir.glob(".*.tmp")) + \
                         list((sf_dir / "plots").glob(f"{d.name}_{OUT_NAME}.png")) + \
                         list((sf_dir / "plots").glob(f"{d.name}_{LEGACY_PNG}")):
                    f.unlink()
                    n += 1
            for f in logdir.glob("sf_log_rank*.jsonl"):
                f.unlink()
            print(f"restart: removed {n} files across {len(objdirs)} objects", flush=True)
        scope = (f"{len(objdirs)} of {n_total} objects (--limit {a.limit})"
                 if a.limit is not None else f"{n_total} objects")
        print(f"{scope}, {size} ranks, cs={'on' if use_cs else 'off'}, "
              f"plots={'on' if a.plots else 'off'}, max_magerr={a.max_magerr}",
              flush=True)

    comm.Barrier()          # the only collective; nothing after this point

    logf = logdir / f"sf_log_rank{rank:03d}.jsonl"
    n_done = n_ok = 0
    for i in range(rank, len(objdirs), size):
        d = objdirs[i]
        sentinel = done_pkl(d)
        if sentinel is not None and sentinel.exists():
            continue
        rec = sf_one(d, use_cs, a.min_epochs, a.timeout, a.plots, a.max_magerr)
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_done += 1
        n_ok += rec["valid"]
        if not rec["valid"]:
            print(f"[rank {rank}] FAIL {d.name}: {rec['fail_reason']}", flush=True)

    print(f"[rank {rank}] {n_ok}/{n_done} ok", flush=True)


if __name__ == "__main__":
    main()