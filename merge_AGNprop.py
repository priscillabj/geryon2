#!/usr/bin/env python
"""
End-to-end: merge linmix fit pkls -> crossmatch AGN properties -> plot one band
by AGN classification.

  1. glob  ~/results/partials/*/*z[gri]_merged.parquet_10cs_linmixfit.pkl
  2. build df: gamma, A_365 + asymmetric (min/max) errors, band, RA
  3. crossmatch via AGNProp df functions -> ... clasf (=Type_105)
  4. save linmix_merged.parquet (pre-crossmatch) and linmix_merged_props.parquet
  5. select one band, scatter A_365 vs gamma coloured by classification

Serial, seconds. AGNProp.py must be importable (same dir or on PYTHONPATH).
"""
import glob, os, pickle, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import AGNProp

PAT        = os.path.expanduser("~/results/partials/*/*z[gri]_merged.parquet_10cs_linmixfit.pkl")
OUT_MERGED = os.path.expanduser("~/results/linmix_merged.parquet")
OUT_PROPS  = os.path.expanduser("~/results/linmix_merged_props.parquet")
OUT_PNG    = os.path.expanduser("~/results/amp_vs_gamma_byclasf.png")

# --- SET THESE to your catalogue paths on geryon2 -------------------------
# data:      columns  wise_ra, wise_dec, AGN ('TRUE'/'FALSE' strings), bat_index, ctpt_name
CATALOG = os.path.expanduser("~/DR2-105monthcatalog-105m_all.csv")
# table_mbh: SEMICOLON-delimited; has "14-150 Lum" (renamed to lx below)
MBH     = os.path.expanduser("~/DR2_DR3_Best MBH - DR2_final_use_this.csv")

# --- plot config: band + classification groups ----------------------------
BAND = "g"                         # <-- band to plot: 'g', 'r', or 'i'
# reconcile these with the printed clasf.value_counts() labels before trusting the plot:
TP1 = ["Sy1", "Sy1.2"]             # Type 1        -> black, hollow markers
SY  = ["Sy1.5"]                    # intermediate  -> C2
TP2 = ["Sy1.8", "Sy1.9", "Sy2"]   # Type 2        -> C1
# --------------------------------------------------------------------------

band_re = re.compile(r"z([gri])_merged")

def val(d, *names):
    for n in names:
        v = d.get(n)
        if v is not None:
            return v
    return np.nan


# --- 1/2. merge --------------------------------------------------------------
rows, files = [], sorted(glob.glob(PAT))
printed = False
for f in files:
    try:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
    except Exception as e:
        print(f"skip (load fail) {f}: {e}"); continue
    if not isinstance(d, dict):
        print(f"skip (not dict) {f}"); continue
    if not printed:
        print("dict keys:", sorted(d.keys())); printed = True   # verify real key names
    m = band_re.search(os.path.basename(f))
    rows.append({
        "source":       os.path.basename(os.path.dirname(f)),
        "band":         m.group(1) if m else "?",
        "RA":           val(d, "RA"),
        "gamma":        val(d, "_gamma_spl"),
        "gamma_minerr": val(d, "_gamma_minerr_spl"),
        "gamma_maxerr": val(d, "_gamma_maxerr_spl"),
        "A_365":        val(d, "_A_365_spl"),
        "A_365_minerr": val(d, "_A_minerr_spl"),
        "A_365_maxerr": val(d, "_A_maxerr_spl"),
        "valid":        bool(d.get("valid", True)),
        "file":         f,
    })

df = pd.DataFrame(rows)
print(f"loaded {len(df)} fits from {len(files)} files")
os.makedirs(os.path.dirname(OUT_MERGED), exist_ok=True)
df.to_parquet(OUT_MERGED, index=False)     # pre-crossmatch, so merge isn't lost if catalogues fail
print("wrote", OUT_MERGED)

# --- 3. crossmatch AGN properties -------------------------------------------
data      = pd.read_csv(CATALOG)
table_mbh = pd.read_csv(MBH, sep=";").rename(columns={"14-150 Lum": "lx"})
_need = ["BAT_ID", "Best_M_BH", "Edd_rat", "L_bol", "lx", "zbest", "NH", "Type_105"]
_missing = [c for c in _need if c not in table_mbh.columns]
if _missing:
    raise SystemExit(f"MBH file {MBH} missing {_missing}; has {list(table_mbh.columns)}")
df = AGNProp.LC2VarFeatdf(df, data)
df = AGNProp.AGNPropdf(df, table_mbh)

# silent all-NaN clasf almost always = bat_index dtype mismatch (object vs int)
print(f"rows={len(df)}  matched_bat_index={df['bat_index'].notna().sum()}  "
      f"has_clasf={df['clasf'].notna().sum()}")
print(df["clasf"].value_counts(dropna=False))   # <- use these labels to set TP1/SY/TP2

df.to_parquet(OUT_PROPS, index=False)
print("wrote", OUT_PROPS)

# # --- 4. plot: one band, coloured by classification --------------------------
# sub = df[(df["band"] == BAND) &
#          np.isfinite(df["gamma"]) & np.isfinite(df["A_365"]) & df["valid"]]
# print(f"plotting band z{BAND}: {len(sub)} sources")

# fig, ax = plt.subplots(figsize=(7, 5))
# for types, color, label in [(TP1, "black", "Type 1"),
#                             (SY,  "C2",    "Sy1.5"),
#                             (TP2, "C1",    "Type 2")]:
#     agn = sub[sub["clasf"].isin(types)]
#     if len(agn) == 0:
#         continue
#     # comment these two lines out to reproduce the example exactly (no error bars):
#     xerr = np.nan_to_num(np.abs(np.vstack([agn["gamma_minerr"], agn["gamma_maxerr"]])))
#     yerr = np.nan_to_num(np.abs(np.vstack([agn["A_365_minerr"], agn["A_365_maxerr"]])))
#     kw = dict(fmt="o", alpha=0.3, color=color, label=f"{label} (n={len(agn)})",
#               xerr=xerr, yerr=yerr, elinewidth=0.6, capsize=0)
#     if color == "black":
#         kw.update(markerfacecolor="none", ecolor="k")
#     ax.errorbar(agn["gamma"], agn["A_365"], **kw)

# ax.set_xlabel(r"$\gamma$  (SF slope)")
# ax.set_ylabel(r"$A_{365}$")
# ax.tick_params(axis='both', labelsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(5e-3,1e0)
# ax.set_ylim(5e-3,1e0)
# ax.grid(True, alpha=0.3)
# ax.set_title(f"z{BAND}")
# ax.legend()
# fig.tight_layout()
# fig.savefig(OUT_PNG, dpi=150)
# print("wrote", OUT_PNG)