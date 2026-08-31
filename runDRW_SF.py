#!/usr/bin/env python3
"""
DRW (Exp kernel) MCMC fits with eztaox, compared against the empirical SF.

Target epoch selection replicates optSF() exactly (newSF.py ~L770):
    MERR < 0.5, MAGLIM > 20.5, SEEING < 3, then modal qid/ccdid/field.
Anything else and the DRW fit and the SF describe different light curves.

Empirical SF + fits are read from the stage-3 partials tree rather than matched
on sky position:
    ~/results/partials/{ra}_{dec}/{basename}_{x}cs.pkl        (SF arrays)
    ~/results/partials/{ra}_{dec}/{basename}_{x}cs*fit.pkl    (SPL/BPL params)
A source with no SF pkl is treated as "not fit" and skipped.

MPI round-robin, no collectives in the work loop. Per-source atomic pkl output;
existing outputs are skipped on resume. Runs without mpirun (size 1) for tests.

NOTE: needs AVX. The Geryon2 login node (Xeon E5620) cannot import jax 0.4.31.
    qsub -I -l nodes=1:ppn=4 -l walltime=01:00:00
    source ~/.bashrc && conda activate eztaox
    python -u run_drw_sf.py --list ~/results/input_files.json --limit 2 --plots
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
import traceback
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
from eztaox.models import UniVarModel
from eztaox.fitter import random_search
from eztaox.kernel_stat2 import gpStat2

from astropy.coordinates import SkyCoord

from mpi4py import MPI

from VarTools import radec_filename, _find_target_obj
from newSF import broken_power_law_flat

BAT_ROOT = Path(os.environ["HOME"]) / "BAT_results"
PARTIALS_ROOT = Path(os.environ["HOME"]) / "results" / "partials"


# ----------------------------------------------------------------- numpyro model
# zero_mean=False + mean_func=None => UniVarModel._default_mean_func reads
# params["mean"], so "mean" is a live fitted parameter -- do not remove it.
# has_jitter=True additionally requires params["log_jitter"].

def make_model(t, y, yerr, has_jitter):
    """Kernel init values are placeholders; real params arrive via .sample()."""
    m = UniVarModel(t, y, yerr, Exp(scale=100.0, sigma=1.0),
                    zero_mean=False, has_jitter=has_jitter)
    # UniVarModel swallows unknown kwargs silently -- zero_men=False would give
    # zero_mean=True with no error -- so verify what actually landed.
    v = vars(m)
    assert v["zero_mean"] is False, v["zero_mean"]
    assert v["has_jitter"] is has_jitter, v["has_jitter"]
    return m


def build_model(t, y, yerr, has_jitter):
    """
    The UniVarModel is built ONCE here, outside the numpyro model function.
    Constructing it inside numpyro_model puts it under NUTS tracing, and its
    __init__ calls jnp.unique(), which needs concrete values -> the
    ConcretizationTypeError. Only .sample() may run inside the traced function.
    """
    m = make_model(t, y, yerr, has_jitter)

    def _kernel_params():
        log_drw_scale = numpyro.sample("log_drw_scale",
                                       dist.Uniform(jnp.log(0.01), jnp.log(1000.0)))
        log_drw_sigma = numpyro.sample("log_drw_sigma",
                                       dist.Uniform(jnp.log(0.01), jnp.log(10.0)))
        log_kernel_param = jnp.stack([log_drw_scale, log_drw_sigma])
        numpyro.deterministic("log_kernel_param", log_kernel_param)
        return log_kernel_param

    def initSampler():
        p = {"log_kernel_param": _kernel_params(),
             "mean": numpyro.sample("mean", dist.Uniform(-0.2, 0.2))}
        if has_jitter:
            p["log_jitter"] = numpyro.sample(
                "log_jitter", dist.Uniform(jnp.log(1e-4), jnp.log(1.0)))
        return p

    def numpyro_model():
        # Normal prior on mean: better NUTS geometry than the uniform above
        p = {"log_kernel_param": _kernel_params(),
             "mean": numpyro.sample("mean", dist.Normal(0.0, 0.1))}
        if has_jitter:
            p["log_jitter"] = numpyro.sample(
                "log_jitter", dist.Uniform(jnp.log(1e-4), jnp.log(1.0)))
        m.sample(p)

    return m, initSampler, numpyro_model


# ------------------------------------------------------------------ file listing

def select_files(args):
    """Mirrors runZMAD.select_files: a JSON list may span bands, so filter."""
    if args.list:
        names = json.loads(Path(os.path.expanduser(args.list)).read_text())
        files = [f if os.path.isabs(f) else str(BAT_ROOT / f) for f in names]
        if args.band:
            files = [f for f in files
                     if re.search(rf"_z{args.band}_", os.path.basename(f))]
    else:
        files = sorted(glob.glob(str(BAT_ROOT / f"*_z{args.band}_merged.parquet")))
    if args.limit:
        files = files[args.offset:args.offset + args.limit]
    return files


# ------------------------------------------------------------------- light curve

def load_target_lc(path, mag_col="MAG_4_TOT_AB", err_col="MERR_4_TOT_AB"):
    """
    Replicates the optSF() target selection exactly. Any divergence here makes
    the DRW/SF comparison apples-to-oranges.
    """
    ra, dec = radec_filename(os.path.basename(path))
    tgt_obj = _find_target_obj(Path(path), SkyCoord(ra, dec, unit="deg"))
    if tgt_obj is None:
        return None, "no_target_match"

    df = pd.read_parquet(path)
    z = df[df["object_index"] == tgt_obj]
    if mag_col not in z.columns:
        return None, "no_mag_column"

    mask = (z[err_col] < 0.5) & (z["MAGLIM"] > 20.5) & (z["SEEING"] < 3)
    if "qid" in z.columns:
        mask &= ((z["qid"] == z["qid"].mode().iloc[0]) &
                 (z["ccdid"] == z["ccdid"].mode().iloc[0]) &
                 (z["field"] == z["field"].mode().iloc[0]))
    z = z[mask]
    if len(z) == 0:
        return None, "empty_target"

    z = z.sort_values("OBSMJD")
    t = z["OBSMJD"].values - z["OBSMJD"].values[0]
    y = z[mag_col].values
    return (t, y - y.mean(), z[err_col].values), None


# ------------------------------------------------------------------ empirical SF

def load_master(path):
    """
    Fit params from bat_master.parquet, keyed on lc_file (= parquet basename),
    the same key merge_AGNprop.py builds by stripping _{x}cs_(linmix|bpl)fit.pkl.

    Only _spl/_bpl columns are kept, so the ZMAD and BASS/clasf columns (and the
    many-to-many row inflation on that side of the merge) cannot leak in here.
    """
    if path is None:
        return None
    df = pd.read_parquet(os.path.expanduser(path))
    df = df[["lc_file"] + [c for c in df.columns if c.endswith(("_spl", "_bpl"))]]
    n = len(df)
    df = df.drop_duplicates("lc_file")
    if len(df) != n:
        print(f"[master] dropped {n - len(df)} duplicate lc_file rows", flush=True)
    return df.set_index("lc_file")


def load_sf_entry(path, x, partials_root, master=None):
    """
    SF arrays always come from the optSF pkl (no DEC, no fit keys in it).
    SPL/BPL params come from bat_master if given, else the *fit.pkl beside it.
    """
    ra, dec = radec_filename(os.path.basename(path))
    d = Path(partials_root) / f"{ra}_{dec}"
    base = os.path.basename(path)
    sf_pkl = d / f"{base}_{x}cs.pkl"
    if not sf_pkl.exists():
        return None
    with open(sf_pkl, "rb") as f:
        entry = dict(pickle.load(f))

    if master is not None:
        if base in master.index:
            entry.update(master.loc[base].dropna().to_dict())
    else:
        for fit in sorted(d.glob(f"{base}_{x}cs*fit.pkl")):   # _linmixfit/_bplfit
            with open(fit, "rb") as f:
                got = pickle.load(f)
            if isinstance(got, dict):
                entry.update(got)
    entry.setdefault("DEC", dec)
    return entry


def empirical_sf(e):
    """
    Pure-data version of newSF.plot_SF -- the committed plot_SF returns None and
    writes a PNG, so it cannot be used from a batch driver.
    Same bin selection: 1 <= dt < 365, SF != 0, dropna.
    """
    ii = pd.IntervalIndex(e["SF"].index)
    sf = e["SF"][(ii.left >= 1) & (ii.right < 365) & (e["SF"] != 0)].dropna()
    dt = np.asarray(sf.index.categories[sf.index.codes].mid, float)
    uperr = e["SFmaxerr"][e["SFmaxerr"].index.isin(sf.index)]
    loerr = e["SFminerr"][e["SFminerr"].index.isin(sf.index)]

    spl = bpl = None
    if "A_365_spl" in e and "gamma_spl" in e:
        spl = e["A_365_spl"] * (dt / 365.0) ** e["gamma_spl"]
    if all(k in e for k in ("A_1_bpl", "gamma_bpl", "dt_break_bpl")):
        bpl = broken_power_law_flat(dt, e["A_1_bpl"], e["gamma_bpl"],
                                    e["dt_break_bpl"])
    return sf, uperr, loerr, spl, bpl, dt


# ----------------------------------------------------------------------- helpers

def seed_for(ra, dec, band):
    """Deterministic per-source seed: resume-stable, core-count independent."""
    return int(hashlib.md5(f"{ra}_{dec}_{band}".encode()).hexdigest()[:8], 16)


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


# --------------------------------------------------------------------- core work

def fit_drw(t, y, yerr, seed, args):
    m, initSampler, numpyro_model = build_model(t, y, yerr, args.jitter)

    init_strategy = init_to_median
    if args.mle_init:
        # eztaox 0.2.0 replaced jaxopt/SLSQP with optax gradient descent
        # (n_opt_step=1000 per candidate) -- keep n_search small.
        bestP, _ = random_search(m, initSampler, jax.random.PRNGKey(seed),
                                 args.n_search, args.n_best)
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
    # data is closed over by numpyro_model, so run() takes only the key
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def process(path, args, master=None):
    ra, dec, band = radec_filename(os.path.basename(path), band=True)
    out = os.path.join(args.outdir, f"drw_{ra}_{dec}_z{band}.pkl")
    if os.path.exists(out) and not args.force:
        return "skip"

    def bail(reason):
        atomic_pickle({"RA": ra, "DEC": dec, "band": band, "valid": False,
                       "fail_reason": reason}, out)
        return reason

    sf_entry = load_sf_entry(path, args.x, args.partials_root, master)
    if sf_entry is None:
        return bail("no_SF_pkl")

    lc, err = load_target_lc(path)
    if lc is None:
        return bail(err)
    t, y, yerr = lc
    if len(t) < args.min_epochs:
        return bail(f"n_epochs={len(t)}")

    seed = seed_for(ra, dec, band)
    mcmc = fit_drw(t, y, yerr, seed, args)

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    chained = mcmc.get_samples(group_by_chain=True)
    summary = summarize(samples)

    tau = float(np.exp(summary["log_drw_scale"]["median"]))
    amp = float(np.exp(summary["log_drw_sigma"]["median"]))
    gp2 = gpStat2(Exp(scale=tau, sigma=amp))

    SFdata, uperr, loerr, spl, bpl, dt = empirical_sf(sf_entry)
    if len(dt) == 0:
        return bail("empty_SF_bins")
    ts = np.logspace(0, 4, args.n_grid)

    # SF for posterior draws, thinned, reduced to a band -- never store 5000 curves
    log_draws = samples["log_kernel_param"][:: args.thin]
    mcmc_sf = np.asarray(jax.vmap(gp2.sf, in_axes=(None, 0))(ts, jnp.exp(log_draws)))
    q = np.percentile(mcmc_sf, [15.865, 50, 84.135], axis=0)

    rhat, ess = {}, {}
    for k, v in chained.items():
        v = np.asarray(v)
        if v.ndim < 2:
            continue
        if args.chains > 1:              # split_gelman_rubin needs >= 2 chains
            rhat[k] = float(np.max(split_gelman_rubin(v)))
        ess[k] = float(np.min(effective_sample_size(v)))

    entry = {
        "RA": ra, "DEC": dec, "band": band, "valid": True,
        "n_epochs": int(len(t)), "baseline": float(t[-1] - t[0]), "seed": seed,
        "has_jitter": bool(args.jitter), "mag": sf_entry.get("mag"),
        # empirical
        "SFdata": SFdata, "SFuperr": uperr, "SFloerr": loerr,
        "spl_yfit": spl, "bpl_yfit": bpl, "dt": dt,
        "sf_fit": {k: v for k, v in sf_entry.items()
                   if k.endswith(("_spl", "_bpl"))},
        # model
        "tau_med": tau, "amp_med": amp,
        "DRW_SF_dt": np.asarray(gp2.sf(dt)),   # median-param SF on empirical bins
        "ts": ts, "DRW_SF_ts": np.asarray(gp2.sf(ts)),
        "mcmc_sf_p16": q[0], "mcmc_sf_p50": q[1], "mcmc_sf_p84": q[2],
        "summary": summary,
        "log_kernel_draws": log_draws.astype(np.float32),
        "rhat": rhat, "ess": ess,
    }
    atomic_pickle(entry, out)

    if args.plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(dt, SFdata, yerr=(loerr, uperr), fmt="o", c="k", capsize=3,
                    label="empirical SF")
        if spl is not None:
            ax.plot(dt, spl, ls="--", label="single power law")
        if bpl is not None:
            ax.plot(dt, bpl, ls=":", label="broken power law")
        ax.plot(ts, gp2.sf(ts), c="tab:green", lw=2, label="DRW (posterior median)")
        ax.fill_between(ts, q[0], q[2], color="tab:green", alpha=0.25)
        ax.set(xscale="log", yscale="log", xlabel="Time diff (days)",
               ylabel="Structure Function (mag)")
        ax.legend(fontsize=9)
        fig.savefig(os.path.join(args.outdir, f"sf_{ra}_{dec}_z{band}.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    return "ok"


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", default=None,
                   help="JSON list of parquet names (relative to ~/BAT_results)")
    p.add_argument("--outdir",
                   default=str(Path(os.environ["HOME"]) / "results" / "drw_sf"))
    p.add_argument("--partials-root", default=str(PARTIALS_ROOT))
    p.add_argument("--master", default=None,
                   help="bat_master.parquet; take SPL/BPL params from there "
                        "instead of the per-source *fit.pkl")
    p.add_argument("--band", default="g")
    p.add_argument("--x", type=int, default=10,
                   help="calstar count in the SF pkl name")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--thin", type=int, default=25)
    p.add_argument("--n-grid", type=int, default=50)
    p.add_argument("--jitter", action="store_true",
                   help="fit log_jitter added in quadrature to yerr; ZTF forced-phot "
                        "errors are often underestimated and that scatter otherwise "
                        "inflates sigma")
    p.add_argument("--mle-init", action="store_true")
    p.add_argument("--n-search", type=int, default=500)
    p.add_argument("--n-best", type=int, default=10)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    comm.Barrier()        # only collective, before the work loop

    files = select_files(args)
    master = load_master(args.master)
    if rank == 0:
        print(f"{len(files)} light curves, {size} ranks, band={args.band}, "
              f"jitter={args.jitter}", flush=True)

    for i in range(rank, len(files), size):     # round-robin, no collectives
        f = files[i]
        try:
            status = process(f, args, master)
            print(f"[rank {rank}] {i} {os.path.basename(f)} {status}", flush=True)
        except Exception:
            print(f"[rank {rank}] {i} {os.path.basename(f)} FAILED\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)

    print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()