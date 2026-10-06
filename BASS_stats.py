#!/usr/bin/env python
"""
Sample-characterisation histograms, styled like plot_SFpar.py --hist:
final sample (solid), variable subsample (dashed), BAT 105-month (grey, AGN
properties only).

  python plot_sample.py --master ~/results/bat_master_zmadflux2.parquet \
      --overlay '~/results/bat_master_zmadflux_sigma>10.parquet' --band g --fit_model spl

AGN properties (LogM_BH, z, Lbol, LogRedd) are counted once per bat_index.
LC properties (median mag, epochs, cadence, baseline) are counted per lc_file,
the same grain as the gamma histogram. Epochs/cadence/baseline are recomputed
from ~/BAT_results/<lc_file> with optSF's target selection and quality mask,
and cached in ~/results/lc_stats.parquet (--rebuild to recompute).
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from plot_SFpar import load, BAND_COLOR, RESULTS     # sets the Agg backend
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from VarTools import radec_filename, _find_target_obj

MBH     = os.path.expanduser("~/DR2_DR3_Best MBH - DR2_final_use_this.csv")
LC_DIR  = Path(os.path.expanduser("~/BAT_results"))
LC_STATS = os.path.join(RESULTS, "lc_stats.parquet")
MAG     = "MAG_4_TOT_AB"        # optSF defaults
MAG_ERR = "MERR_4_TOT_AB"

# name: (master column, 105-month column, xlabel)
AGN_Q = {
    "mbh":  ("LogM_BH", "Best_M_BH", r"$\log\,M_{\rm BH}\ [M_\odot]$"),
    "z":    ("z",       "zbest",     "redshift"),
    "lbol": ("Lbol",    "L_bol",     r"$\log\,L_{\rm bol}\ [\rm erg\,s^{-1}]$"),
    "edd":  ("LogRedd", "Edd_rat",   r"$\log\,L/L_{\rm Edd}$"),
}
LC_Q = {
    "mag":      ("med_mag",    "median magnitude"),
    "epochs":   ("n_epochs",   "number of epochs"),
    "cadence":  ("cad_median", "median cadence [days, distinct nights]"),
    "baseline": ("baseline",   "LC length [days]"),
}


def lc_row(name):
    """optSF target selection: match, then quality mask with field/ccd/qid
    modes taken over all target rows *before* the mask (as in optSF)."""
    path = LC_DIR / name
    ra, dec = radec_filename(name)
    obj = _find_target_obj(path, SkyCoord(ra, dec, unit="deg"))
    if obj is None:
        return {"lc_file": name}
    cols = [c for c in ["object_index", "OBSMJD", MAG, MAG_ERR, "MAGLIM", "SEEING",
                        "qid", "ccdid", "field"] if c in pq.read_schema(path).names]
    t = pd.read_parquet(path, columns=cols)
    t = t[t["object_index"] == obj]
    m = (t[MAG_ERR] < 0.5) & (t["MAGLIM"] > 20.5) & (t["SEEING"] < 3)
    if "qid" in t.columns:
        for c in ("qid", "ccdid", "field"):
            m &= t[c] == t[c].mode().iloc[0]
    mjd = t.loc[m, "OBSMJD"].values
    nights = np.unique(np.floor(mjd))       # Palomar nights don't cross MJD boundaries
    return {"lc_file": name, "med_mag": t.loc[m, MAG].median(), "n_epochs": len(mjd),
            "cad_median": np.median(np.diff(nights)) if len(nights) > 1 else np.nan,
            "baseline": np.ptp(mjd) if len(mjd) else np.nan}


def lc_stats(names, rebuild):
    cache = (pd.read_parquet(LC_STATS) if os.path.exists(LC_STATS) and not rebuild
             else pd.DataFrame(columns=["lc_file"]))
    todo = sorted(set(names) - set(cache["lc_file"]))
    if todo:
        print(f"lc_stats: computing {len(todo)} light curves ({len(cache)} cached)")
        cache = pd.concat([cache, pd.DataFrame([lc_row(n) for n in todo])], ignore_index=True)
        cache.to_parquet(LC_STATS + ".tmp", index=False)
        os.replace(LC_STATS + ".tmp", LC_STATS)
    return cache.set_index("lc_file")


def as_log(v, label):
    """Lbol may be stored linear or log10; decide from the data and say so."""
    v = pd.to_numeric(v, errors="coerce")
    if np.nanmedian(v) > 100:
        print(f"  {label}: linear values, taking log10")
        return np.log10(v)
    return v


def per_source(df):
    n0 = len(df)
    df = df.dropna(subset=["bat_index"]).drop_duplicates("bat_index")
    print(f"  {n0} rows -> {len(df)} unique bat_index")
    return df


def hist(name, xlabel, sub, osub, ref, band, fit_model):
    sub, osub = sub[np.isfinite(sub)], osub[np.isfinite(osub)]
    pool = [sub, osub] + ([ref[np.isfinite(ref)]] if ref is not None else [])
    bins = np.histogram_bin_edges(np.concatenate(pool), bins="auto")
    fig, ax = plt.subplots(figsize=(6, 4))
    if ref is not None:
        ref = pool[2]
        ax.hist(ref, bins=bins,histtype="step", edgecolor="0.5",
                label=f"BAT 105-month (n={len(ref)})")
    ax.hist(sub, bins=bins, histtype="step", linewidth=1.8, edgecolor=BAND_COLOR[band],
            label=f"final sample (n={len(sub)})")
    ax.hist(osub, bins=bins, histtype="step", linewidth=2.7, edgecolor="#B89CFF",
            linestyle="--",
            label=f"variable subsample (n={len(osub)}, {100 * len(osub) / len(sub):.0f}%)")
    ax.set_ylim(top=1.4 * ax.get_ylim()[1])
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    if name == "z":
        ax.set_xlim(left = -0.01 ,right=0.5)
    elif name == "cadence":
        ax.set_xlim(left = -0.05 ,right=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    # ax.set_title(f"z{band}")
    out = os.path.join(RESULTS, f"{band}band_{name}_distribution_{fit_model}_ovl.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def hist_bands(name, xlabel, data, fit_model):
    """One edge-only line per band (final sample), shared bins."""
    data = {b: v[np.isfinite(v)] for b, v in data.items()}
    bins = np.histogram_bin_edges(np.concatenate(list(data.values())), bins="auto")
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, v in data.items():
        ax.hist(v, bins=bins, histtype="step", linewidth=1.8, edgecolor=BAND_COLOR[b],
                label=f"z{b} (n={len(v)})")
    ax.set_ylim(top=1.4 * ax.get_ylim()[1])
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    if name == "cadence":
        ax.set_xlim(left = -0.05 ,right=20)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, f"gri_{name}_distribution_{fit_model}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--master", default=os.path.expanduser("~/results/bat_master_zmadflux2.parquet"))
    p.add_argument("--overlay", required=True, help="variable-subsample parquet")
    p.add_argument("--band", default="g", choices=["g", "r", "i"])
    p.add_argument("--fit_model", default="spl", choices=["spl", "bpl"])
    p.add_argument("--rebuild", action="store_true", help="recompute lc_stats cache")
    p.add_argument("--allbands", action="store_true",
                   help="also plot LC properties of the final sample in g, r and i")
    a = p.parse_args()

    sub  = load(os.path.expanduser(a.master), a.band, a.fit_model)
    osub = load(os.path.expanduser(a.overlay), a.band, a.fit_model)

    cat = pd.read_csv(MBH).replace("#VALUE!", np.nan)
    print("105-month:"); cat = per_source(cat.rename(columns={"BAT_ID": "bat_index"}))
    print("final:");     src  = per_source(sub)
    print("variable:");  osrc = per_source(osub)
    for name, (col, ccol, xlabel) in AGN_Q.items():
        ref = pd.to_numeric(cat[ccol], errors="coerce")
        if name == "edd":
            ref = np.log10(ref.where(ref > 0))
        s, o = pd.to_numeric(src[col], errors="coerce"), pd.to_numeric(osrc[col], errors="coerce")
        if name == "lbol":
            ref, s, o = as_log(ref, "105-month"), as_log(s, "final"), as_log(o, "variable")
        hist(name, xlabel, s.values, o.values, ref.values, a.band, a.fit_model)

    st = lc_stats(set(sub["lc_file"]) | set(osub["lc_file"]), a.rebuild)
    for d in (sub, osub):
        for c in ("med_mag", "n_epochs", "cad_median", "baseline"):
            d[c] = d["lc_file"].map(st[c])
    print(f"lc_stats: no target/epochs for {sub['n_epochs'].isna().sum()} of {len(sub)} lc_files")
    for name, (col, xlabel) in LC_Q.items():
        hist(name, xlabel, sub[col].astype(float).values,
             osub[col].astype(float).values, None, a.band, a.fit_model)

    if a.allbands:
        subs = {b: load(os.path.expanduser(a.master), b, a.fit_model) for b in "gri"}
        st = lc_stats(set().union(*(s["lc_file"] for s in subs.values())), False)
        for name, (col, xlabel) in LC_Q.items():
            hist_bands(name, xlabel, {b: s["lc_file"].map(st[col]).astype(float).values
                                      for b, s in subs.items()}, a.fit_model)