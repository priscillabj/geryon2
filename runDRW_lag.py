#!/usr/bin/env python3
"""
Multiband DRW fits with interband lag (eztaox MultiVarModel, g+r).

One fit per SOURCE, not per file: the g and r parquets are paired by the
{ra}_{dec} prefix of their basenames. Both bands share a single time origin
(the earliest OBSMJD across the pair) -- the lag is meaningless otherwise.

Target epoch selection replicates optSF() (newSF.py ~L770):
    MERR < 0.5, MAGLIM > 20.5, SEEING < 3, then modal qid/ccdid/field.

MPI round-robin, no collectives in the work loop. Per-source atomic pkl output;
existing outputs are skipped on resume. Runs without mpirun (size 1) for tests.

NOTE: needs AVX. The Geryon2 login node (Xeon E5620) cannot import jax 0.4.31.
    qsub -I -l nodes=1:ppn=4 -l walltime=01:00:00
    source ~/.bashrc && conda activate eztaox
    python -u run_drw_lag.py --list ~/results/input_files.json --limit 2 --plots
"""
import os

# Thread pinning MUST precede the jax import or every rank spawns a full
# XLA/BLAS thread pool and the ranks thrash each other.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=1 "
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import argparse
import glob
import hashlib
import json
import pickle
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import split_gelman_rubin, effective_sample_size
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value

from eztaox.kernels.quasisep import Exp
from eztaox.models import MultiVarModel
from eztaox.fitter import random_search
from eztaox.ts_utils import formatlc

from astropy.coordinates import SkyCoord

from mpi4py import MPI

# NOT `from newSF import ...`: importing newSF pulls in corner -> arviz, which
# writes a daily-warning stamp via write-temp-then-rename. Every rank races on
# the same temp path and the losers die with FileNotFoundError.
from VarTools import radec_filename, _find_target_obj

BAT_ROOT = Path(os.environ["HOME"]) / "BAT_results"
BANDS = ("g", "r")
BAND_INDEX = {"g": 0, "r": 1}


# ----------------------------------------------------------------- numpyro model

def make_model(x, y, yerr, zero_mean):
    m = MultiVarModel(x, y, yerr, Exp(scale=100.0, sigma=1.0), len(BANDS),
                      has_lag=True, zero_mean=zero_mean)
    # MultiVarModel swallows unknown kwargs silently, so verify what landed
    v = vars(m)
    assert v["zero_mean"] is zero_mean, v["zero_mean"]
    assert v["has_lag"] is True, v["has_lag"]
    return m


def build_model(x, y, yerr, zero_mean, lag_max):
    """
    The MultiVarModel is built ONCE, outside the numpyro model function.
    Building it inside puts it under NUTS tracing and its __init__ calls
    jnp.unique(), which needs concrete values -> ConcretizationTypeError.
    Only .sample() may run inside the traced function.
    """
    m = make_model(x, y, yerr, zero_mean)

    def _params():
        log_drw_scale = numpyro.sample("log_drw_scale",
                                       dist.Uniform(jnp.log(0.01), jnp.log(1000.0)))
        log_drw_sigma = numpyro.sample("log_drw_sigma",
                                       dist.Uniform(jnp.log(0.01), jnp.log(10.0)))
        log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
        numpyro.deterministic("log_kernel_param", log_kernel_param)
        p = {
            "log_kernel_param": log_kernel_param,
            # r-band sigma = g-band sigma * exp(log_amp_scale)
            "log_amp_scale": numpyro.sample("log_amp_scale", dist.Uniform(-2, 2)),
            "lag": numpyro.sample("lag", dist.Uniform(-lag_max, lag_max)),
        }
        # With zero_mean=True, MultiVarModel.get_mean returns 0.0 and never
        # reads params["mean"] -- sampling it would just echo the prior back
        # and cost tree depth, so it is only sampled when it is actually used.
        if not zero_mean:
            p["mean"] = numpyro.sample(
                "mean", dist.Normal(jnp.zeros(len(BANDS)), 0.1))
        return p

    def initSampler():
        return _params()

    def numpyro_model():
        m.sample(_params())

    return m, initSampler, numpyro_model


# ------------------------------------------------------------------ file listing

def pair_files(args):
    """
    Group the file list into {(ra, dec): {band: path}} and keep only sources
    with every band present. Returns a sorted list of (ra, dec, {band: path}).
    """
    if args.list:
        names = json.loads(Path(os.path.expanduser(args.list)).read_text())
        files = [f if os.path.isabs(f) else str(BAT_ROOT / f) for f in names]
    else:
        files = sorted(glob.glob(str(BAT_ROOT / "*_merged.parquet")))

    # Match any z<band> file, merged or per-CCD, as runZMAD.select_files does.
    # Requiring "_merged" would silently drop every source that only exists as
    # per-CCD exposures (e.g. *_000258_zg_ccd05_q3.parquet).
    groups = defaultdict(lambda: defaultdict(list))
    for f in files:
        b = os.path.basename(f)
        mb = re.search(r"_z([gri])[_.]", b)
        if not mb or mb.group(1) not in BANDS:
            continue
        ra, dec = radec_filename(b)
        groups[(ra, dec)][mb.group(1)].append(f)

    def pick(paths):
        """Prefer the merged light curve; else the per-CCD file with most rows."""
        merged = [p for p in paths if "_merged." in os.path.basename(p)]
        if merged:
            return sorted(merged)[0]
        if args.merged_only:
            return None
        return max(paths, key=lambda p: os.path.getsize(p))

    out, n_ccd_only = [], 0
    for (ra, dec), d in sorted(groups.items()):
        chosen = {b: pick(d[b]) for b in BANDS if d.get(b)}
        if any(v is None for v in chosen.values()):
            continue
        if len(chosen) < len(BANDS):
            continue
        if any("_merged." not in os.path.basename(v) for v in chosen.values()):
            n_ccd_only += 1
        out.append((ra, dec, chosen))

    if args.limit:
        out = out[args.offset:args.offset + args.limit]
    return out, len(groups), n_ccd_only


# ------------------------------------------------------------------- light curve

def load_band(path, mag_col="MAG_4_TOT_AB", err_col="MERR_4_TOT_AB"):
    """Same target selection as optSF(); returns raw OBSMJD, not an offset."""
    ra, dec = radec_filename(os.path.basename(path))
    obj = _find_target_obj(Path(path), SkyCoord(ra, dec, unit="deg"))
    if obj is None:
        return None, "no_target_match"

    z = pd.read_parquet(path)
    z = z[z["object_index"] == obj]
    if mag_col not in z.columns:
        return None, "no_mag_column"

    mask = (z[err_col] < 0.5) & (z["MAGLIM"] > 20.5) & (z["SEEING"] < 3)
    if "qid" in z.columns:
        mask &= ((z["qid"] == z["qid"].mode().iloc[0]) &
                 (z["ccdid"] == z["ccdid"].mode().iloc[0]) &
                 (z["field"] == z["field"].mode().iloc[0]))
    z = z[mask]
    if len(z) == 0:
        return None, "empty_after_cuts"

    z = z.sort_values("OBSMJD")
    return (z["OBSMJD"].values, z[mag_col].values, z[err_col].values), None


def load_pair(paths, min_epochs):
    """
    Build the formatlc inputs for one source.

    Both bands are offset by the SAME t0 (earliest epoch across the pair). The
    notebook used the g-band first epoch, which breaks if g is missing; any
    per-band origin would absorb the lag into the time axis.
    Per-band means are subtracted so the inter-band offset does not have to be
    carried by the mean or amp_scale parameters.
    """
    raw = {}
    for b, p in paths.items():
        lc, err = load_band(p)
        if lc is None:
            return None, f"{b}:{err}"
        if len(lc[0]) < min_epochs:
            return None, f"{b}:n_epochs={len(lc[0])}"
        raw[b] = lc

    t0 = min(raw[b][0][0] for b in BANDS)
    ts = {b: raw[b][0] - t0 for b in BANDS}
    ys = {b: raw[b][1] - raw[b][1].mean() for b in BANDS}
    yerrs = {b: raw[b][2] for b in BANDS}
    nep = {b: int(len(raw[b][0])) for b in BANDS}
    baseline = float(max(ts[b][-1] for b in BANDS))
    return (ts, ys, yerrs, nep, baseline), None


# ----------------------------------------------------------------------- helpers

def seed_for(ra, dec):
    return int(hashlib.md5(f"{ra}_{dec}_gr".encode()).hexdigest()[:8], 16)


def atomic_pickle(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def summarize(samples):
    out = {}
    for k, v in samples.items():
        v = np.asarray(v)
        med = np.median(v, axis=0)
        lo = np.percentile(v, 15.865, axis=0)
        hi = np.percentile(v, 84.135, axis=0)
        out[k] = {"mean": np.mean(v, axis=0), "std": np.std(v, axis=0),
                  "median": med, "p15.865": lo, "p84.135": hi,
                  "uperr": hi - med, "loerr": med - lo}
    return out


def rail_fraction(lag_draws, lag_max, frac=0.05):
    """
    Share of the lag posterior within 5% of either hard prior bound.

    Uniform(-lag_max, lag_max) is a hard wall: an unconstrained lag piles up
    against it and still yields a finite median, so a histogram of medians can
    show structure that is pure prior. Cut on this before interpreting lags.
    """
    edge = lag_max * (1 - frac)
    return float(np.mean(np.abs(lag_draws) > edge))


# --------------------------------------------------------------------- core work

def fit_lag(x, y, yerr, seed, args):
    m, initSampler, numpyro_model = build_model(x, y, yerr, args.zero_mean,
                                                args.lag_max)
    init_strategy = init_to_median
    if args.mle_init:
        # eztaox 0.2.0 replaced jaxopt/SLSQP with optax gradient descent
        # (n_opt_step=1000 per candidate) -- keep n_search small.
        bestP, _ = random_search(m, initSampler, jax.random.PRNGKey(seed),
                                 args.n_search, args.n_best)
        init_strategy = init_to_value(values={
            "log_drw_scale": bestP["log_kernel_param"][0],
            "log_drw_sigma": bestP["log_kernel_param"][1],
            "log_amp_scale": bestP["log_amp_scale"],
            "lag": bestP["lag"],
        })

    mcmc = MCMC(
        NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9,
             init_strategy=init_strategy),
        num_warmup=args.warmup, num_samples=args.samples,
        num_chains=args.chains, chain_method="sequential", progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))   # data is closed over
    return mcmc


def process(ra, dec, paths, args):
    out = os.path.join(args.outdir, f"drwlag_{ra}_{dec}.pkl")
    if os.path.exists(out) and not args.force:
        return "skip"

    def bail(reason):
        atomic_pickle({"RA": ra, "DEC": dec, "valid": False,
                       "fail_reason": reason}, out)
        return reason

    got, err = load_pair(paths, args.min_epochs)
    if got is None:
        return bail(err)
    ts, ys, yerrs, nep, baseline = got

    x, y, yerr = formatlc(ts, ys, yerrs, BAND_INDEX)
    seed = seed_for(ra, dec)
    mcmc = fit_lag(x, y, yerr, seed, args)

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    chained = mcmc.get_samples(group_by_chain=True)
    summary = summarize(samples)

    rhat, ess = {}, {}
    for k, v in chained.items():
        v = np.asarray(v)
        if v.ndim < 2:
            continue
        if args.chains > 1:        # split_gelman_rubin needs >= 2 chains
            rhat[k] = float(np.max(split_gelman_rubin(v)))
        ess[k] = float(np.min(effective_sample_size(v)))

    lag_draws = samples["lag"]
    entry = {
        "RA": ra, "DEC": dec, "bands": list(BANDS), "valid": True,
        "n_epochs": nep, "baseline": baseline, "seed": seed,
        "zero_mean": bool(args.zero_mean), "lag_max": args.lag_max,
        "tau_med": float(np.exp(summary["log_drw_scale"]["median"])),
        "amp_med": float(np.exp(summary["log_drw_sigma"]["median"])),
        # r-band sigma = g-band sigma * exp(log_amp_scale)
        "amp_scale_med": float(np.exp(summary["log_amp_scale"]["median"])),
        "lag_med": float(summary["lag"]["median"]),
        "lag_rail_frac": rail_fraction(lag_draws, args.lag_max),
        "summary": summary,
        "lag_draws": lag_draws[:: args.thin].astype(np.float32),
        "rhat": rhat, "ess": ess,
    }
    atomic_pickle(entry, out)

    if args.plots:
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.5),
                                     gridspec_kw={"width_ratios": [2, 1]})
        for b, c in zip(BANDS, ("mediumseagreen", "firebrick")):
            a0.errorbar(ts[b], ys[b], yerr=yerrs[b], fmt=".", c=c, ms=4,
                        alpha=0.7, label=f"{b}-band")
        a0.invert_yaxis()
        a0.set(xlabel="MJD - t0 (days)", ylabel="mag - <mag>")
        a0.legend(fontsize=9)

        a1.hist(lag_draws, bins=60, color="C0", alpha=0.6)
        a1.axvline(entry["lag_med"], c="k", lw=2)
        for s in (-1, 1):
            a1.axvline(s * args.lag_max, c="r", ls="--", lw=1)
        a1.set(xlabel="lag (days)",
               title=f"lag={entry['lag_med']:.2f} d, "
                     f"rail={entry['lag_rail_frac']:.2f}")
        fig.suptitle(f"{ra} {dec}   "
                     rf"$\tau$={entry['tau_med']:.0f} d, "
                     rf"$\sigma_g$={entry['amp_med']:.3f}, "
                     rf"amp$_r/_g$={entry['amp_scale_med']:.2f}")
        fig.savefig(os.path.join(args.outdir, f"drwlag_{ra}_{dec}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    return "ok"


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", default=None,
                   help="JSON list of parquet names (relative to ~/BAT_results)")
    p.add_argument("--outdir",
                   default=str(Path(os.environ["HOME"]) / "results" / "drw_lag"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--merged-only", action="store_true",
                   help="skip sources whose band has no *_merged file, instead "
                        "of falling back to the largest per-CCD file")
    p.add_argument("--lag-max", type=float, default=10.0,
                   help="hard bound on the Uniform lag prior, in days")
    p.add_argument("--zero-mean", action="store_true",
                   help="assume zero-mean GP; otherwise a per-band mean is fitted")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--mle-init", action="store_true")
    p.add_argument("--n-search", type=int, default=500)
    p.add_argument("--n-best", type=int, default=5)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    job_start = time.time()

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    comm.Barrier()      # only collective, before the work loop

    sources, n_all, n_ccd = pair_files(args)
    if rank == 0:
        print(f"{len(sources)} sources with all of {BANDS} "
              f"(of {n_all} with any band; {n_ccd} use a per-CCD file for at "
              f"least one band), {size} ranks, lag_max={args.lag_max}, "
              f"zero_mean={args.zero_mean}", flush=True)

    for i in range(rank, len(sources), size):    # round-robin, no collectives
        ra, dec, paths = sources[i]
        t0 = time.time()
        try:
            status = process(ra, dec, paths, args)
            print(f"[rank {rank}] {i} {ra}_{dec} {status} "
                  f"in {time.time()-t0:.1f}s", flush=True)
        except Exception:
            print(f"[rank {rank}] {i} {ra}_{dec} FAILED\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)

    h, rem = divmod(time.time() - job_start, 3600)
    m, s = divmod(rem, 60)
    print(f"[rank {rank}] done. walltime {int(h):02d}h {int(m):02d}m {s:04.1f}s",
          flush=True)


if __name__ == "__main__":
    main()