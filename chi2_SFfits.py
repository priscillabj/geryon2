#!/usr/bin/env python
"""
Goodness-of-fit chi^2 for every SF fit in the master table, appended as columns.

    python -u chi2_SFfits.py --master ~/results/bat_master_zmadflux.parquet
    python -u chi2_SFfits.py --master ... --out ~/results/bat_master_chi2.parquet --png chi2.png

Data points are read from the BASE SF pkl (~/results/partials/{source}/{lc_file}_{x}cs.pkl),
not the fit pkls, so SPL and BPL are scored on exactly the same points. The window
reproduces SF_linmix / bpl_mcmc: bins with left >= 1, right <= 365, SF != 0, dropna.

Model predictions come from the master-table parameters. broken_power_law_flat with
A_1_bpl is used for both BPL variants ('flat' and 'at_break'), since at_break's
A_break = A_1 * dt_break**gamma -- the curve is identical.

Two chi^2 per model, from the same residuals:
  chi2      linear space, sigma = (SFmaxerr + SFminerr)/2   -- bpl_mcmc's own likelihood
  chi2_log  log10 space,  sigma = sigma_lin / (SF * ln10)   -- closer to SF_linmix's

Neither is exactly SF_linmix's statistic: linmix also uses x-errors and an intrinsic
scatter term. And with the current (suppressed) SF errors the normalisation is wrong,
so treat rchi2 as a RANKING, not a goodness-of-fit test, until bootstrap errors land.
"""
import argparse, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARTIALS = Path(os.environ["HOME"]) / "results" / "partials"
NPAR = {"spl": 2, "bpl": 3}          # linmix also fits intrinsic scatter -> arguably 3
PARAMS = {"spl": ["A_1_spl", "gamma_spl"],
          "bpl": ["A_1_bpl", "gamma_bpl", "dt_break_bpl"]}


def load_sf(source, lc_file, x):
    """Base SF dict. Handles both payload shapes MPI_linmixfit.py tolerates."""
    p = PARTIALS / source / f"{lc_file}_{x}cs.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    return d["SF_dict"][0] if isinstance(d, dict) and "SF_dict" in d else d


def fit_window(d, maxerr_col, minerr_col):
    """(dt, sf, sigma) for the bins the fitters actually used."""
    sf = d["SF"]
    ii = pd.IntervalIndex(sf.index)
    sf = sf[(ii.left >= 1) & (ii.right <= 365) & (sf != 0)].dropna()
    if len(sf) == 0:
        return None
    # IntervalIndex(...) rather than .categories[.codes]: works whether the SF
    # index is the CategoricalIndex pd.cut leaves behind or a bare IntervalIndex
    dt = np.asarray(pd.IntervalIndex(sf.index).mid, dtype=float)
    sig = ((d[maxerr_col].reindex(sf.index) + d[minerr_col].reindex(sf.index)) / 2)
    return dt, sf.to_numpy(dtype=float), sig.to_numpy(dtype=float)


def predict(model, dt, r):
    if model == "spl":
        return r["A_1_spl"] * dt ** r["gamma_spl"]
    return np.where(dt <= r["dt_break_bpl"],
                    r["A_1_bpl"] * dt ** r["gamma_bpl"],
                    r["A_1_bpl"] * r["dt_break_bpl"] ** r["gamma_bpl"])


def chi2_row(sf, sig, pred, npar):
    ok = np.isfinite(sf) & np.isfinite(pred) & np.isfinite(sig) & (sig > 0) & (sf > 0) & (pred > 0)
    n, dof = int(ok.sum()), int(ok.sum()) - npar
    out = {"n": n, "dof": dof}
    spaces = {"":     (sf[ok], pred[ok], sig[ok]),
              "_log": (np.log10(sf[ok]), np.log10(pred[ok]), sig[ok] / (sf[ok] * np.log(10)))}
    for tag, (y, mu, s) in spaces.items():
        c = float(np.sum(((y - mu) / s) ** 2))
        out[f"chi2{tag}"] = c
        out[f"rchi2{tag}"] = c / dof if dof > 0 else np.nan
        out[f"pval{tag}"] = float(chi2_dist.sf(c, dof)) if dof > 0 else np.nan
    return out


def compute(df, args):
    rows = []
    n_nopkl = n_nowin = 0
    for r in df.itertuples(index=False):
        rec = {"lc_file": r.lc_file}
        d = load_sf(r.source, r.lc_file, args.x)
        if d is None:
            n_nopkl += 1
            rows.append(rec); continue
        w = fit_window(d, args.maxerr_col, args.minerr_col)
        if w is None:
            n_nowin += 1
            rows.append(rec); continue
        dt, sf, sig = w
        rec["n_sf"] = len(sf)
        for model, pars in PARAMS.items():
            p = {k: getattr(r, k, np.nan) for k in pars}
            if not all(np.isfinite(v) for v in p.values()):
                continue
            st = chi2_row(sf, sig, predict(model, dt, p), NPAR[model])
            rec.update({f"{k}_{model}": v for k, v in st.items() if k != "n"})
        rows.append(rec)
    print(f"{n_nopkl} rows with no base SF pkl, {n_nowin} with no points in [1,365]")
    return pd.DataFrame(rows)


def summarise(df):
    for model in PARAMS:
        for tag, name in [("", "linear"), ("_log", "log")]:
            v = df.get(f"rchi2{tag}_{model}")
            if v is None or not v.notna().any():
                continue
            q = v.dropna().quantile([.16, .5, .84]).round(2).tolist()
            frac = (v.dropna() > 5).mean()
            print(f"{model} rchi2 ({name:6s}) 16/50/84 = {q},  frac>5 = {frac:.2f},  "
                  f"n = {v.notna().sum()}")


def plot(df, png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tag, name in zip(axes, ["", "_log"], ["linear space", "log space"]):
        for model, c in [("spl", "C0"), ("bpl", "C1")]:
            v = df.get(f"rchi2{tag}_{model}")
            if v is None:
                continue
            v = v[np.isfinite(v) & (v > 0)]
            if not len(v):
                continue
            ax.hist(np.log10(v), bins=40, histtype="step", lw=1.8, color=c,
                    label=f"{model.upper()}  N={len(v)}  med={np.median(v):.1f}")
        ax.axvline(0, color="k", ls=":", lw=1, label=r"$\chi^2_\nu=1$")
        ax.set_xlabel(r"$\log_{10}\ \chi^2_\nu$")
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("light curves")
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=os.path.expanduser("~/results/bat_master_zmadflux.parquet"))
    p.add_argument("--out", default=None, help="default: <master>_chi2.parquet")
    p.add_argument("--png", default=os.path.expanduser("~/results/chi2_dist.png"))
    p.add_argument("--x", type=int, default=10, help="calstar count in the pkl name")
    p.add_argument("--maxerr-col", default="SFmaxerr", help="repoint at bootstrap errors")
    p.add_argument("--minerr-col", default="SFminerr")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="print the chi2 table and summary, write nothing (no parquet, no png)")
    args = p.parse_args()

    df = pd.read_parquet(args.master)
    for c in ("lc_file", "source"):
        if c not in df.columns:
            raise SystemExit(f"{args.master} has no {c!r} column; got {list(df.columns)[:20]}")
    if df["lc_file"].duplicated().any():
        raise SystemExit("lc_file is not unique in the master -- fix before merging chi2 on it")
    if args.limit:
        df = df.head(args.limit)

    ch = compute(df, args)

    if args.dry_run:
        cols = [c for c in ["lc_file", "band", "n_sf",
                            "dof_spl", "rchi2_spl", "rchi2_log_spl", "pval_log_spl",
                            "dof_bpl", "rchi2_bpl", "rchi2_log_bpl", "pval_log_bpl"]
                if c in ch.columns or c in df.columns]
        tbl = df[["lc_file"] + [c for c in cols if c in df.columns and c != "lc_file"]] \
                .merge(ch, on="lc_file", how="left", validate="one_to_one")[cols]
        print(tbl.to_string(max_rows=40, float_format=lambda v: f"{v:.3g}"))
        summarise(ch)
        print("\n--dry-run: nothing written")
        return

    out = df.merge(ch, on="lc_file", how="left", validate="one_to_one")
    dest = args.out or args.master.replace(".parquet", "_chi2.parquet")
    out.to_parquet(dest, index=False)
    print(f"wrote {dest}  ({len(out)} rows, +{len(ch.columns) - 1} columns)")
    summarise(out)
    plot(out, args.png)


if __name__ == "__main__":
    main()