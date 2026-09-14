#!/usr/bin/env python
"""
Find SF fits whose slope is hostage to one discrepant bin.

    python -u sf_outliers.py --master ~/results/bat_master_zmadflux2.parquet --dry-run --limit 20
    python -u sf_outliers.py --master ~/results/bat_master_zmadflux2.parquet

Motivation: a bin that disagrees with its neighbours drags the SPL fit toward
itself, so its chi2 residual UNDERSTATES how bad it is. Both statistics here are
computed without reference to the stored fit, and without the SF error bars
(which are known to be suppressed), so they stay valid pre-bootstrap.

  z_rob   residual from a Theil-Sen line in log10 SF vs log10 dt, scaled by
          1.4826*MAD of the residuals. Theil-Sen is the median of all pairwise
          slopes: it tolerates ~29% contamination, so 2 bad bins out of 8 do not
          move it. A bin with |z_rob| >~ 4 disagrees with the trend the other
          bins define.
  dgamma  jackknife: max |gamma_TS(all) - gamma_TS(drop one bin)|. This is the
          quantity you actually care about -- how much one bin owns the slope.

Points are the same fit window the fitters use (bins in [1,365], SF != 0), read
from the base SF pkl. Nothing is excluded here: this ranks candidates for you to
look at, it does not refit.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chi2_SFfits import load_sf, fit_window     # same window, same pkl convention

MIN_N = 5            # Theil-Sen + MAD on fewer bins is not meaningful
MAD_FLOOR = 0.005    # dex. Below ~1% bin-to-bin scatter nothing is an outlier in
                     # any useful sense, and an unfloored MAD -> 0 turns float
                     # noise into |z| of 10+ on a perfectly clean power law.
BAND_COLOR = {"g": "mediumseagreen", "r": "firebrick", "i": "goldenrod"}


def theil_sen(x, y):
    """(slope, intercept): median of pairwise slopes, median intercept."""
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    good = dx != 0
    if not good.any():
        return np.nan, np.nan
    slope = np.median((y[j] - y[i])[good] / dx[good])
    return slope, np.median(y - slope * x)


def robust_z(x, y):
    """Residuals from the Theil-Sen line, scaled by a robust sigma."""
    m, b = theil_sen(x, y)
    if not np.isfinite(m):
        return None
    r = y - (m * x + b)
    mad = max(np.median(np.abs(r - np.median(r))) * 1.4826, MAD_FLOOR)
    return m, r / mad


def audit(dt, sf):
    x, y = np.log10(dt), np.log10(sf)
    rz = robust_z(x, y)
    if rz is None:
        return {}
    gamma, z = rz

    # jackknife: how much does the slope move if one bin goes away
    dg = np.array([abs(gamma - theil_sen(np.delete(x, i), np.delete(y, i))[0])
                   for i in range(len(x))])

    k = int(np.argmax(np.abs(z)))
    kg = int(np.argmax(dg))
    return {"ts_gamma":   float(gamma),
            "max_z_rob":  float(np.abs(z).max()),      # how discrepant the worst bin is. Above ~4 means that bin disagrees with the other bins. Below ~3, nothing is weird.
            "dt_max_z":   float(dt[k]),                # tells where the bad bin is
            "sign_max_z": int(np.sign(z[k])),          # tells you which way (+1 = the bin sits above the fit)
            "n_out":      int((np.abs(z) > 4).sum()),
            "dgamma":     float(dg.max()),             # how much the slope moves if you drop that bin.Above ~0.1 means the slope is hostage to it.
            "dt_dgamma":  float(dt[kg])}



def compute(df, args):
    rows, keep = [], {}
    for r in df.itertuples(index=False):
        rec = {"lc_file": r.lc_file}
        d = load_sf(r.source, r.lc_file, args.x)
        w = None if d is None else fit_window(d, "SFmaxerr", "SFminerr")
        if w is not None:
            dt, sf, sig = w
            rec["n_sf"] = len(sf)
            if len(sf) >= MIN_N:
                rec.update(audit(dt, sf))
                keep[r.lc_file] = (dt, sf, sig, getattr(r, "band", "?"),
                                   getattr(r, "A_1_spl", np.nan),
                                   getattr(r, "gamma_spl", np.nan))
        rows.append(rec)
    return pd.DataFrame(rows), keep


def gallery(top, keep, png):
    n = len(top)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax, r in zip(axes.flat, top.itertuples(index=False)):
        dt, sf, sig, band, a_spl, g_spl = keep[r.lc_file]
        x, y = np.log10(dt), np.log10(sf)
        m, b = theil_sen(x, y)
        ax.errorbar(dt, sf, yerr=sig, fmt="o", ms=4,
                    color=BAND_COLOR.get(band, "grey"), elinewidth=0.8, capsize=0)
        ax.plot(dt, 10 ** (m * x + b), "-", color="k", lw=1,
                label=rf"Theil-Sen  $\gamma$={m:.2f}")
        if np.isfinite(a_spl) and np.isfinite(g_spl):
            ax.plot(dt, a_spl * dt ** g_spl, "--", color="crimson", lw=1.2,
                    label=rf"linmix  $\gamma$={g_spl:.2f}")
        bad = np.isclose(dt, r.dt_max_z)
        ax.plot(dt[bad], sf[bad], "o", ms=12, mfc="none", mec="crimson", mew=1.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{r.lc_file.split('_z')[0]} {band}\n"
                     rf"$z$={r.max_z_rob:.1f} at {r.dt_max_z:.0f}d, "
                     rf"$\Delta\gamma$={r.dgamma:.2f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.supxlabel("Time diff (days)"); fig.supylabel("Structure Function (mag)")
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")


def hist_dt(ch, png, zmin, minn):
    """Where along the lag axis do discrepant bins sit, and on which side.

    Two cuts matter. Restricting to |z| > zmin keeps the histogram from being
    filled by the worst bin of an otherwise clean fit (every fit has one, and its
    lag is arbitrary). Restricting to n_sf >= minn drops the 5-6 point fits,
    where Theil-Sen and MAD both rest on too few residuals to mean much."""
    s = ch.dropna(subset=["max_z_rob", "dt_max_z"])
    s = s[(s.max_z_rob > zmin) & (s.n_sf >= minn)]
    if not len(s):
        print(f"hist: nothing with |z|>{zmin} and n_sf>={minn}")
        return
    bins = np.logspace(0, np.log10(365), 25)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sgn, color, lab in [(1, "crimson", "above trend"),
                            (-1, "steelblue", "below trend")]:
        v = s.loc[s.sign_max_z == sgn, "dt_max_z"]
        if len(v):
            ax.hist(v, bins=bins, histtype="step", lw=1.8, color=color,
                    label=f"{lab}  (n={len(v)})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$ of the discrepant bin (days)")
    ax.set_ylabel("light curves")
    ax.set_title(rf"$|z_{{\rm rob}}|>{zmin}$, $n_{{\rm sf}}\geq{minn}$   "
                 f"({len(s)} of {len(ch)})", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")

    lo = s[s.dt_max_z < 3]
    print(f"  {len(lo)}/{len(s)} sit below 3 d "
          f"({(lo.sign_max_z == 1).sum()} above trend, "
          f"{(lo.sign_max_z == -1).sum()} below)")


def scatter(ch, png):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    s = ch.dropna(subset=["max_z_rob", "dt_max_z"])
    sc = ax.scatter(s.dt_max_z, s.max_z_rob, c=s.dgamma, s=18,
                    cmap="viridis", norm=matplotlib.colors.LogNorm())
    ax.axhline(4, color="crimson", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta t$ of worst bin (days)")
    ax.set_ylabel(r"$|z_{\rm rob}|$ of worst bin")
    fig.colorbar(sc, label=r"max $|\Delta\gamma|$ (jackknife)")
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")



def hist_dgamma(ch, png, zmin, minn):
    """How much slope is at stake, over the flagged fits.

    Same two cuts as hist_dt: |z| > zmin keeps out the worst bin of an otherwise
    clean fit, n_sf >= minn keeps out the small-sample cases where dgamma just
    measures leverage. Split by sign so you can see whether high or low bins do
    more damage."""
    s = ch.dropna(subset=["max_z_rob", "dgamma"])
    s = s[(s.max_z_rob > zmin) & (s.n_sf >= minn) & (s.dgamma > 0)]
    if not len(s):
        print(f"hist: nothing with |z|>{zmin} and n_sf>={minn}")
        return
    bins = np.logspace(np.log10(s.dgamma.min()), np.log10(s.dgamma.max()), 25)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for sgn, color, lab in [(1, "crimson", "bin above trend"),
                            (-1, "steelblue", "bin below trend")]:
        v = s.loc[s.sign_max_z == sgn, "dgamma"]
        if len(v):
            ax.hist(v, bins=bins, histtype="step", lw=1.8, color=color,
                    label=f"{lab}  (n={len(v)})")
    ax.axvline(0.1, color="k", ls=":", lw=1, label=r"$\Delta\gamma=0.1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"max $|\Delta\gamma|$ from dropping one bin")
    ax.set_ylabel("light curves")
    ax.set_title(rf"$|z_{{\rm rob}}|>{zmin}$, $n_{{\rm sf}}\geq{minn}$   "
                 f"({len(s)} of {len(ch)})", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"  {(s.dgamma > 0.1).sum()}/{len(s)} move gamma by >0.1, "
          f"{(s.dgamma > 0.3).sum()} by >0.3")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master", default=os.path.expanduser("~/results/bat_master_zmadflux2.parquet"))
    p.add_argument("--out", default=None)
    p.add_argument("--png", default=os.path.expanduser("~/results/sf_outliers.png"))
    p.add_argument("--x", type=int, default=10)
    p.add_argument("--top", type=int, default=12, help="worst N to draw in the gallery")
    p.add_argument("--zmin", type=float, default=4.0,
                   help="|z_rob| above which a bin counts as discrepant")
    p.add_argument("--min-n", type=int, default=10,
                   help="minimum n_sf for the ranked table and the histogram; "
                        "below ~10, dgamma just measures small-sample leverage")
    p.add_argument("--dgamma", type=float, default=0.0,
               help="minimum max|dgamma| for the ranked table and gallery; "
                    "0 = no cut")
    
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    df = pd.read_parquet(args.master)
    if args.limit:
        df = df.head(args.limit)

    extra = [c for c in ("band", "A_1_spl", "gamma_spl") if c in df.columns]
    keys = df[["source", "lc_file"] + extra].drop_duplicates("lc_file")
    ch, kept = compute(keys, args)

    ok = ch.dropna(subset=["max_z_rob"])
    print(f"{len(ok)} of {len(ch)} light curves audited (need n_sf >= {MIN_N})")
    if len(ok):
        print(f"|z_rob| 50/84/97.5 = "
              f"{ok.max_z_rob.quantile([.5,.84,.975]).round(2).tolist()};  "
              f"{(ok.max_z_rob > 4).sum()} with |z|>4;  "
              f"{((ok.max_z_rob > 4) & (ok.dgamma > 0.1)).sum()} of those also move "
              f"gamma by >0.1")
        cand = ok[(ok.max_z_rob > args.zmin) & (ok.n_sf >= args.min_n)
                  & (ok.dgamma > args.dgamma)]
        top = cand.sort_values("dgamma", ascending=False).head(args.top)
        print(f"\n{len(cand)} candidates (|z|>{args.zmin}, n_sf>={args.min_n}), "
              f"dgamma>{args.dgamma}),"
              f"worst by slope impact:")
        print(top[["lc_file", "n_sf", "ts_gamma", "max_z_rob", "dt_max_z",
                   "sign_max_z", "dgamma", "dt_dgamma"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    dest = args.out or args.master.replace(".parquet", "_outliers.parquet")
    df.merge(ch, on="lc_file", how="left", validate="many_to_one") \
      .to_parquet(dest, index=False)
    print(f"wrote {dest}")
    if len(ok):
        scatter(ok, args.png)
        hist_dt(ok, args.png.replace(".png", "_dthist.png"), args.zmin, args.min_n)
        hist_dgamma(ok, args.png.replace(".png", "_dghist.png"), args.zmin, args.min_n)
        if len(top):
            gallery(top, kept, args.png.replace(".png", f"dgamma>{args.dgamma}_gallery.png"))


if __name__ == "__main__":
    main()