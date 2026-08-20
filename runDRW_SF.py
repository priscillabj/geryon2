#!/usr/bin/env python3
"""
DRW (Exp kernel) MCMC fits with eztaox + comparison against empirical SF.

MPI round-robin, one source per rank at a time, no collectives in the work loop.
Per-source pkl output with atomic write; existing outputs are skipped on resume.

Usage:
  mpirun -np 80 python run_drw_sf.py \
      --lc-glob "$HOME/BAT_results/*.parquet" \
      --sf-dict "$HOME/results/SF_dict.pkl" \
      --outdir "$HOME/results/drw_sf" \
      --band g --format parquet
"""
import os

# Thread pinning MUST happen before jax is imported, otherwise every rank
# spawns a full XLA/BLAS thread pool and 40 ranks/node thrash.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "XLA_FLAGS"):
    os.environ.setdefault(_v, "1")
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 "
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import argparse
import glob
import hashlib
import pickle
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import arviz as az
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value
from numpyro.handlers import seed as numpyro_seed

from eztaox.kernels.quasisep import Exp
from eztaox.models import UniVarModel
from eztaox.fitter import random_search
from eztaox.stats import gpStat2          # <-- CONFIRM this import path

from astropy.coordinates import SkyCoord
import astropy.units as u

from mpi4py import MPI

# your modules
from VarTools import radec_filename
from newSF import plot_SF               # <-- CONFIRM which module plot_SF lives in


# ----------------------------------------------------------------- numpyro model

def initSampler():
    log_drw_scale = numpyro.sample("log_drw_scale",
                                   dist.Uniform(jnp.log(0.01), jnp.log(1000.0)))
    log_drw_sigma = numpyro.sample("log_drw_sigma",
                                   dist.Uniform(jnp.log(0.01), jnp.log(10.0)))
    log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
    numpyro.deterministic("log_kernel_param", log_kernel_param)
    mean = numpyro.sample("mean", dist.Uniform(low=-0.2, high=0.2))
    return {"log_kernel_param": log_kernel_param, "mean": mean}


def numpyro_model(t, yerr, y=None):
    log_drw_scale = numpyro.sample("log_drw_scale",
                                   dist.Uniform(jnp.log(0.01), jnp.log(1000.0)))
    log_drw_sigma = numpyro.sample("log_drw_sigma",
                                   dist.Uniform(jnp.log(0.01), jnp.log(10.0)))
    log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
    numpyro.deterministic("log_kernel_param", log_kernel_param)
    mean = numpyro.sample("mean", dist.Normal(0.0, 0.1))

    k = Exp(scale=100.0, sigma=1.0)   # init values unused; params come from sample_params
    m = UniVarModel(t, y, yerr, k, zero_mean=False)
    m.sample({"log_kernel_param": log_kernel_param, "mean": mean})


# ----------------------------------------------------------------------- loaders

def load_lc_csv(path, band):
    """corrlc_*.csv layout from the local version."""
    df = pd.read_csv(path)
    b = df[df["filter"] == f"ZTF_{band}"]
    if len(b) == 0:
        return None
    b = b.sort_values("mjd")
    t = b["mjd"].values - b["mjd"].values[0]
    y = b["magtot_clr_corr"].values
    return t, y - y.mean(), b["magunc_clr_corr"].values


def load_lc_parquet(path, band, mag_col="MAG_4_TOT_AB", err_col="MERR_4_TOT_AB"):
    """
    BAT_results parquet: AGN target + calibration stars in one file.
    Target must be picked by coordinate match, NOT object_index == 0.
    """
    from VarTools import _find_target_obj, quality_flags, quality_cuts
    df = pd.read_parquet(path)
    ra, dec = radec_filename(path)
    obj = _find_target_obj(path, df=df, verbose=False)
    df = df[df["object_index"] == obj]
    # df = quality_cuts(quality_flags(df), mag_col)   # <-- CONFIRM signature / `x` arg
    mask = df[err_col] < 0.5

    if 'qid' in df.columns:
        qid    = df['qid'].mode().iloc[0]
        ccdid  = df['ccdid'].mode().iloc[0]
        field  = df['field'].mode().iloc[0]
        mask &= (
        (df['qid']   == qid)  &
        (df['ccdid'] == ccdid) &
        (df['field'] == field)
    )

    df = df[mask]

    if len(df) == 0:
        return None
    df = df.sort_values("mjd")
    t = df["mjd"].values - df["mjd"].values[0]
    y = df[mag_col].values
    return t, y - y.mean(), df[err_col].values


# ----------------------------------------------------------------------- helpers

def seed_for(ra, dec, band):
    """Deterministic per-source seed: resume-stable and core-count independent."""
    h = hashlib.md5(f"{ra}_{dec}_{band}".encode()).hexdigest()
    return int(h[:8], 16)


def build_sf_lookup(SF_dict, tol_arcsec=1.0):
    ras = np.array([float(d["RA"]) for d in SF_dict])
    has_dec = "DEC" in SF_dict[0]
    if has_dec:
        cat = SkyCoord(ras * u.deg, np.array([float(d["DEC"]) for d in SF_dict]) * u.deg)

    def lookup(ra, dec):
        if not has_dec:
            j = int(np.argmin(np.abs(ras - ra)))
            return j if abs(ras[j] - ra) * 3600 < tol_arcsec else None
        idx, sep, _ = SkyCoord(ra * u.deg, dec * u.deg).match_to_catalog_sky(cat)
        return int(idx) if float(sep.arcsec) < tol_arcsec else None

    return lookup


def empirical_sf(sf_entry, band):
    """plot_SF draws as a side effect; capture returns and discard the figure."""
    fig = plt.figure()
    try:
        SFdata, uperr, loerr, spl = plot_SF(sf_entry, band, model="single", label=False)
        _, _, _, bpl = plot_SF(sf_entry, band, model="broken", label=False)
    finally:
        plt.close(fig)
        plt.close("all")
    dt = SFdata.index.categories[SFdata.index.codes].mid
    return SFdata, uperr, loerr, spl, bpl, np.asarray(dt, float)


def atomic_pickle(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# --------------------------------------------------------------------- core work

def fit_drw(t, y, yerr, seed, args):
    init_strategy = init_to_median
    if args.mle_init:
        bestP, _ = random_search(
            UniVarModel(t, y, yerr, Exp(scale=100.0, sigma=1.0), zero_mean=False),
            initSampler, jax.random.PRNGKey(seed), args.n_search, args.n_best,
        )
        init_strategy = init_to_value(values={
            "log_drw_scale": bestP["log_kernel_param"][0],
            "log_drw_sigma": bestP["log_kernel_param"][1],
        })

    mcmc = MCMC(
        NUTS(numpyro_model, dense_mass=True, target_accept_prob=0.9,
             init_strategy=init_strategy),
        num_warmup=args.warmup, num_samples=args.samples,
        num_chains=args.chains, chain_method="sequential", progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), t, yerr, y=y)
    return mcmc, az.from_numpyro(mcmc)


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


def process(path, sf_lookup, SF_dict, args):
    ra, dec = radec_filename(path)
    band = args.band
    out = os.path.join(args.outdir, f"drw_{ra}_{dec}_{band}.pkl")
    if os.path.exists(out) and not args.force:
        return "skip"

    sf_id = sf_lookup(ra, dec)
    if sf_id is None:
        atomic_pickle({"RA": ra, "DEC": dec, "band": band, "valid": False,
                       "fail_reason": "not_in_SF_dict"}, out)
        return "not_variable"

    loader = load_lc_parquet if args.format == "parquet" else load_lc_csv
    lc = loader(path, band)
    if lc is None:
        atomic_pickle({"RA": ra, "DEC": dec, "band": band, "valid": False,
                       "fail_reason": f"no_{band}_epochs"}, out)
        return "empty"
    t, y, yerr = lc
    if len(t) < args.min_epochs:
        atomic_pickle({"RA": ra, "DEC": dec, "band": band, "valid": False,
                       "fail_reason": f"n_epochs={len(t)}"}, out)
        return "too_short"

    seed = seed_for(ra, dec, band)
    mcmc, data = fit_drw(t, y, yerr, seed, args)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    summary = summarize(samples)

    tau = float(np.exp(summary["log_drw_scale"]["median"]))
    amp = float(np.exp(summary["log_drw_sigma"]["median"]))
    gp2 = gpStat2(Exp(scale=tau, sigma=amp))

    SFdata, uperr, loerr, spl, bpl, dt = empirical_sf(SF_dict[sf_id], band)
    ts = np.logspace(0, 4, args.n_grid)

    # SF for posterior draws, thinned, then reduced to a band (never store 5000 curves)
    log_draws = data.posterior.stack(sample=["chain", "draw"])["log_kernel_param"].values.T
    log_draws = np.asarray(log_draws)[:: args.thin]
    mcmc_sf = np.asarray(
        jax.vmap(gp2.sf, in_axes=(None, 0))(ts, jnp.exp(log_draws))
    )
    q = np.percentile(mcmc_sf, [15.865, 50, 84.135], axis=0)

    try:
        rhat = {k: float(np.max(v)) for k, v in az.rhat(data).data_vars.items()}
        ess = {k: float(np.min(v)) for k, v in az.ess(data).data_vars.items()}
    except Exception:
        rhat, ess = {}, {}

    entry = {
        "RA": ra, "DEC": dec, "band": band, "valid": True,
        "n_epochs": int(len(t)), "baseline": float(t[-1] - t[0]), "seed": seed,
        # empirical
        "SFdata": SFdata, "SFuperr": uperr, "SFloerr": loerr,
        "spl_yfit": spl, "bpl_yfit": bpl, "dt": dt,
        # model
        "tau_med": tau, "amp_med": amp,
        "DRW_SF_dt": np.asarray(gp2.sf(dt)),        # median-param SF on empirical bins
        "ts": ts, "DRW_SF_ts": np.asarray(gp2.sf(ts)),
        "mcmc_sf_p16": q[0], "mcmc_sf_p50": q[1], "mcmc_sf_p84": q[2],
        "summary": summary,
        "log_kernel_draws": log_draws.astype(np.float32),
        "rhat": rhat, "ess": ess,
    }
    atomic_pickle(entry, out)

    if args.plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(dt, SFdata, yerr=[loerr, uperr], fmt="o", c="k", label="empirical SF")
        ax.plot(dt, spl, ls="--", label="single PL")
        ax.plot(dt, bpl, ls=":", label="broken PL")
        ax.plot(ts, gp2.sf(ts), c="tab:green", lw=2, label="DRW (posterior median)")
        ax.fill_between(ts, q[0], q[2], color="tab:green", alpha=0.25)
        ax.set(xscale="log", yscale="log", xlabel=r"$\Delta t$ [d]", ylabel="SF [mag]")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(args.outdir, f"sf_{ra}_{dec}_{band}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    return "ok"


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lc-glob", required=True)
    p.add_argument("--sf-dict", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    p.add_argument("--band", default="g")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--thin", type=int, default=25)
    p.add_argument("--n-grid", type=int, default=50)
    p.add_argument("--mle-init", action="store_true")
    p.add_argument("--n-search", type=int, default=10_000)
    p.add_argument("--n-best", type=int, default=10)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    comm.Barrier()   # only collective, before the work loop

    files = sorted(glob.glob(os.path.expandvars(os.path.expanduser(args.lc_glob))))
    if args.limit:
        files = files[: args.limit]
    with open(os.path.expanduser(args.sf_dict), "rb") as f:
        SF_dict = pickle.load(f)
    sf_lookup = build_sf_lookup(SF_dict)

    if rank == 0:
        print(f"{len(files)} light curves, {len(SF_dict)} SF entries, {size} ranks",
              flush=True)

    for i in range(rank, len(files), size):        # round-robin, no collectives
        f = files[i]
        try:
            status = process(f, sf_lookup, SF_dict, args)
            print(f"[rank {rank}] {i} {os.path.basename(f)} {status}", flush=True)
        except Exception:
            print(f"[rank {rank}] {i} {os.path.basename(f)} FAILED\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)

    print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()