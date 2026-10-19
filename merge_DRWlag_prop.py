#!/usr/bin/env python3
"""
Merge per-source drwlag_*.pkl into a parquet table, and optionally make the
lag / drw-parameter histograms split by AGN class.

No jax import, so this runs on the login node.

  python merge_drw_lag.py
  python merge_drw_lag.py --master ~/results/bat_master.parquet --hist
"""
import argparse, glob, os, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP1 = ["Sy1", "Sy1.2"]          # <-- CONFIRM: your notebook's tp1/tp2 lists
TP2 = ["Sy1.8", "Sy1.9", "Sy2"]

p = argparse.ArgumentParser()
p.add_argument("--outdir", default=os.path.expandvars("$HOME/results/drw_lag"))
p.add_argument("--master", default=None,
               help="bat_master.parquet, for the clasf column used by --hist")
p.add_argument("--hist", action="store_true")
p.add_argument("--rhat-max", type=float, default=1.05)
p.add_argument("--ess-min", type=float, default=400.0)
p.add_argument("--rail-max", type=float, default=0.10,
               help="reject lags with more than this share of the posterior "
                    "against the prior bound")
a = p.parse_args()

entries, rows = [], []
for f in sorted(glob.glob(os.path.join(a.outdir, "drwlag_*.pkl"))):
    with open(f, "rb") as fh:
        e = pickle.load(fh)
    entries.append(e)
    if not e.get("valid"):
        rows.append({"RA": e["RA"], "DEC": e["DEC"], "valid": False,
                     "fail_reason": e.get("fail_reason")})
        continue
    s = e["summary"]
    rhat = max(e["rhat"].values()) if e["rhat"] else np.nan
    ess = min(e["ess"].values()) if e["ess"] else np.nan
    row = {
        "RA": e["RA"], "DEC": e["DEC"], "valid": True,
        "nepochs_g": e["n_epochs"]["g"], "nepochs_r": e["n_epochs"]["r"],
        "baseline": e["baseline"],
        "log_drw_scale": float(s["log_drw_scale"]["median"]),
        "drw_scale_uperr": float(s["log_drw_scale"]["uperr"]),
        "drw_scale_loerr": float(s["log_drw_scale"]["loerr"]),
        "log_drw_sigma": float(s["log_drw_sigma"]["median"]),
        "sigma_uperr": float(s["log_drw_sigma"]["uperr"]),
        "sigma_loerr": float(s["log_drw_sigma"]["loerr"]),
        "log_amp_scale": float(s["log_amp_scale"]["median"]),
        "amp_uperr": float(s["log_amp_scale"]["uperr"]),
        "amp_loerr": float(s["log_amp_scale"]["loerr"]),
        "lag": float(s["lag"]["median"]),
        "lag_uperr": float(s["lag"]["uperr"]),
        "lag_loerr": float(s["lag"]["loerr"]),
        "lag_rail_frac": e["lag_rail_frac"],
        "tau_med": e["tau_med"], "amp_med": e["amp_med"],
        "amp_scale_med": e["amp_scale_med"],
        "rhat_max": rhat, "ess_min": ess,
        "tau_resolved": bool(e["tau_med"] < 0.4 * e["baseline"]),
        "lag_ok": bool(e["lag_rail_frac"] < a.rail_max),
        "converged": bool((np.isnan(rhat) or rhat < a.rhat_max)
                          and (np.isnan(ess) or ess > a.ess_min)),
        "fail_reason": None,
    }
    rows.append(row)

with open(os.path.join(a.outdir, "drw_lag_merged.pkl"), "wb") as fh:
    pickle.dump(entries, fh, protocol=pickle.HIGHEST_PROTOCOL)
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(a.outdir, "drw_lag_params.parquet"), index=False)

print(df["valid"].value_counts().to_dict())
if df["valid"].any():
    v = df[df.valid]
    print(f"converged    {int(v.converged.sum())}/{len(v)}")
    print(f"tau_resolved {int(v.tau_resolved.sum())}/{len(v)}")
    print(f"lag_ok       {int(v.lag_ok.sum())}/{len(v)}  "
          f"(rail_frac < {a.rail_max})")
if (~df["valid"]).any():
    print(df[~df.valid].fail_reason.value_counts().to_dict())

if not a.hist:
    raise SystemExit

if a.master is None:
    raise SystemExit("--hist needs --master for the clasf column")

# m = pd.read_parquet(os.path.expanduser(a.master))
# key = ["RA", "DEC"] if "DEC" in m.columns else ["RA"]
# m = m[key + ["clasf"]].drop_duplicates(key)
# m = m[["RA", "DEC", "clasf"]].dropna(subset=["RA", "DEC", "clasf"]).drop_duplicates(["RA", "DEC"])# # only sources that actually constrain their parameters
# v = df[df.valid & df.converged & df.lag_ok].merge(m, on=key, how="left")

m = pd.read_csv(os.path.expanduser(a.master))
m = m[["RA", "DEC", "Type_105"]].dropna().rename(columns={"Type_105": "clasf"})
m = m.drop_duplicates(["RA", "DEC"])

from astropy.coordinates import SkyCoord
import astropy.units as u

v = df[df.valid & df.converged & df.lag_ok].copy()
cat = SkyCoord(m.RA.values * u.deg, m.DEC.values * u.deg)
src = SkyCoord(v.RA.values * u.deg, v.DEC.values * u.deg)
idx, sep, _ = src.match_to_catalog_sky(cat)
ok = sep.arcsec < 1.0
v["clasf"] = pd.Series(m.clasf.values[idx], index=v.index).where(ok)
print(f"{len(v)} sources; {int(ok.sum())} matched to a class within 1\"")

print(f"{len(v)} sources into the histograms, "
      f"{int(v.clasf.isna().sum())} without a class")

groups = [(TP2, "#56B4E9", "step",       "-"), 
          (["Sy1.5"], "#E69F00", "stepfilled", "-"),
          (TP1, "black",      "step",       "--")]
kw = dict(histtype="stepfilled", alpha=0.3, ec="k")

fig, ax = plt.subplots(figsize=(7, 5))
for tp, c, ht, ls in groups:
    g = v[v.clasf.isin(tp)]
    if len(g) == 0:
        continue
    bins = np.linspace(v.lag.min(), v.lag.max(), 40)
    ax.hist(g["lag"], bins=bins, color=c, label=f"{tp} (n={len(g)})",
            histtype=ht, linestyle=ls, linewidth=1.8,
            alpha=0.3 if ht == "stepfilled" else 1.0,
            ec=c if ht == "step" else "k")
ax.axvline(0, ls="-.", c="gray")
ax.set_xlabel("lag (days)", fontsize=14)
ax.legend(fontsize=9)
fig.savefig(os.path.join(a.outdir, "drw_lag_hist.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)

par = ["log_drw_sigma", "log_drw_scale"]
fig, axes = plt.subplots(1, len(par), figsize=(16, 6))
for i, j in enumerate(par):
    bins = np.linspace(v[j].min(), v[j].max(), 40)
    for tp, c, ht, ls in groups:
        g = v[v.clasf.isin(tp)]
        if len(g) == 0:
            continue
        axes[i].hist(g[j], bins=bins, color=c, label=f"{tp} (n={len(g)})",
                     histtype=ht, linestyle=ls, linewidth=1.8,
                     alpha=0.3 if ht == "stepfilled" else 1.0,
                     ec=c if ht == "step" else "k")    
    axes[i].set_xlabel(j, fontsize=14)
    axes[i].legend(fontsize=8)

# literature reference values
axes[0].axvline(np.log(0.02 * np.sqrt(KELLY_TAU / 2)), ls="--", c="red", label="Kelly+09")
axes[0].axvline(np.log(0.2 / np.sqrt(2)), ls="--", c="orange", label="MacLeod+10")
axes[0].axvline(-1.050, ls="--", c="brown", label="Jha+25")
axes[1].axvline(1, ls="--", c="black", label="min ts")
axes[1].axvline(4, ls="--", c="red", label="QSO")
axes[1].axvline(3, ls="--", c="orange", label="QSO")
axes[1].axvline(np.log(295), ls="--", c="brown", label="Sy1")
for b in axes:
    b.legend(fontsize=8)
fig.savefig(os.path.join(a.outdir, "drw_lag_parhist.png"), dpi=200,
            bbox_inches="tight")
plt.close(fig)
print("wrote drw_lag_hist.png, drw_lag_parhist.png")