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
import sf_keys                       # single source of truth for fit-key names

# PAT        = os.path.expanduser("~/results/partials/*/*z[gri]_merged.parquet_10cs_linmixfit.pkl")
PAT_SPL    = os.path.expanduser("~/results/partials/*/*z[gri]_merged.parquet_10cs_linmixfit.pkl")
PAT_BPL    = os.path.expanduser("~/results/partials/*/*z[gri]_merged.parquet_10cs_bplfit.pkl")
OUT_MERGED = os.path.expanduser("~/results/SFfit_merged_zmadflux.parquet")
# OUT_PROPS  = os.path.expanduser("~/results/linmix_merged_props.parquet")
OUT_PROPS  = os.path.expanduser("~/results/bat_master_zmadflux.parquet")
# OUT_PNG    = os.path.expanduser("~/results/amp_vs_gamma_byclasf_all.png")

# --- SET THESE to your catalogue paths on geryon2 -------------------------
# data:      columns  wise_ra, wise_dec, AGN ('TRUE'/'FALSE' strings), bat_index, ctpt_name
CATALOG = os.path.expanduser("~/DR2-105monthcatalog-105m_all.csv")
# table_mbh: SEMICOLON-delimited; has "14-150 Lum" (renamed to lx below)
MBH     = os.path.expanduser("~/DR2_DR3_Best MBH - DR2_final_use_this.csv")

# --- plot config: band + classification groups ----------------------------
BAND = "g"                         # <-- band to plot: 'g', 'r', or 'i'
MODEL = "spl"                      # 'spl' (plots A_365) or 'bpl' (plots A_break)

# reconcile these with the printed clasf.value_counts() labels before trusting the plot:
TP1 = ["Sy1", "Sy1.2"]             # Type 1        -> black, hollow markers
SY  = ["Sy1.5"]                    # intermediate  -> C2
TP2 = ["Sy1.8", "Sy1.9", "Sy2"]   # Type 2        -> C1

AMP   = {"spl": "A_365", "bpl": "A_break"}

# -----ZMAD config ------------------------------------------------------------
# ZMAD_PAT      = os.path.expanduser("~/results/zmad_sub_*.parquet")   # picks up _r/_i when they exist
# ZMAD_PAT      = os.path.expanduser("~/results/zmad_subs_g_flux.parquet")   
ZMAD_PAT      = os.path.expanduser("~/results/zmad_all_flux.parquet")   
ZMAD_APERTURE = "4"
ZMAD_METRICS  = ["ok", "fail_reason", "sigma", "perc", "agn", "cs_mean", "cs_std",
                 "cs_median", "cs_max", "n_stars", "n_stars_preclip", "n_epochs", "space"]

band_re = re.compile(r"z([gri])_merged")
pkl_re  = re.compile(r"_\d+cs_(linmix|bpl)fit\.pkl$")
_STRIP = re.compile(r"_(spl|bpl)$")
ID      = ["lc_file", "source", "band", "RA"]

# def val(d, *names):
#     for n in names:
#         v = d.get(n)
#         if v is not None:
#             return v
#     return np.nan

def val(d, name):
    """value for `name`, mapping absent/None -> NaN."""
    v = d.get(name)
    return np.nan if v is None else v

# def extract(d, model):
#     """Pull every canonical fit key for `model` (defined once in sf_keys) into a
#     row, dropping the _spl/_bpl suffix so columns unify across models
#     (gamma_spl/gamma_bpl -> gamma, A_365_spl -> A_365, A_break_bpl -> A_break).
#     Keys live only in sf_keys.py — nothing to update here when they change."""
#     return {_STRIP.sub("", key): val(d, key) for key in sf_keys.keys_for(model)}

def extract(d, model):
    """Canonical fit keys for `model` (defined once in sf_keys), re-tagged with a
    uniform _<model> suffix so nothing collides between models. Keys that already
    carry the suffix are unchanged; ones that don't (e.g. A_1) gain it."""
    out = {}
    for key in sf_keys.keys_for(model):
        base = _STRIP.sub("", key).lstrip("_")
        out[f"{base}_{model}"] = val(d, key)
    return out

def load_zmad(pattern=ZMAD_PAT, aperture=ZMAD_APERTURE):
    """One row per light-curve file, metrics prefixed zmad_, keyed on lc_file.
    Failure rows (aperture is null) are kept so that 'ran and failed' stays
    distinguishable from 'never run'."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no ZMAD parquets match {pattern}")
    frames = []
    for p in paths:
        z = pd.read_parquet(p)
        z = z[(z["aperture"].astype(str) == str(aperture)) | z["aperture"].isna()]
        if not len(z):
            raise SystemExit(f"{p}: no rows at aperture={aperture!r}")
        z["lc_file"] = z["file"].map(os.path.basename)
        # a real aperture-4 row supersedes a stale null-aperture failure row
        z = (z.sort_values("aperture", na_position="last")
               .drop_duplicates("lc_file", keep="first"))
        frames.append(z)
        print(f"zmad: {len(z):5d} rows from {os.path.basename(p)}")
    z = pd.concat(frames, ignore_index=True)
    dup = z["lc_file"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(f"non-unique lc_file across ZMAD parquets ({dup.sum()} rows) "
                         f"-- overlapping band files, or mag and flux runs mixed:\n"
                         f"{z.loc[dup, ['lc_file', 'space', 'agg', 'col']].head(10)}")
    cols = [c for c in ZMAD_METRICS if c in z.columns]
    return z[["lc_file"] + cols].rename(columns={c: f"zmad_{c}" for c in cols})


# --- 1/2. merge --------------------------------------------------------------
# rows, files = [], sorted(glob.glob(PAT))
# printed = False
# for f in files:
#     try:
#         with open(f, "rb") as fh:
#             d = pickle.load(fh)
#     except Exception as e:
#         print(f"skip (load fail) {f}: {e}"); continue
#     if not isinstance(d, dict):
#         print(f"skip (not dict) {f}"); continue
#     if not printed:
#         print("dict keys:", sorted(d.keys())); printed = True   # verify real key names
#     m = band_re.search(os.path.basename(f))
#     rows.append({
#         "source":       os.path.basename(os.path.dirname(f)),
#         "band":         m.group(1) if m else "?",
#         "RA":           val(d, "RA"),
#         "gamma":        val(d, "_gamma_spl"),
#         "gamma_minerr": val(d, "_gamma_minerr_spl"),
#         "gamma_maxerr": val(d, "_gamma_maxerr_spl"),
#         "A_365":        val(d, "_A_365_spl"),
#         "A_365_minerr": val(d, "_A_minerr_spl"),
#         "A_365_maxerr": val(d, "_A_maxerr_spl"),
#         "valid":        bool(d.get("valid", True)),
#         "file":         f,
#     })

# df = pd.DataFrame(rows)
# print(f"loaded {len(df)} fits from {len(files)} files")
# os.makedirs(os.path.dirname(OUT_MERGED), exist_ok=True)
# df.to_parquet(OUT_MERGED, index=False)     # pre-crossmatch, so merge isn't lost if catalogues fail
# print("wrote", OUT_MERGED)

# --- 1/2. merge both models --------------------------------------------------
# rows = []
frames  = {}
printed = set()
for model, pat in (("spl", PAT_SPL), ("bpl", PAT_BPL)):
    # files = sorted(glob.glob(pat))
    rows, files = [], sorted(glob.glob(pat))
    n_ok = 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception as e:
            print(f"skip (load fail) {f}: {e}"); continue
        if not isinstance(d, dict):
            print(f"skip (not dict) {f}"); continue
        if model not in printed:                       # verify real key names once per model
            print(f"[{model}] dict keys:", sorted(d.keys())); printed.add(model)
        m = band_re.search(os.path.basename(f))
        row = {
            "source": os.path.basename(os.path.dirname(f)),
            "band":   m.group(1) if m else "?",
            # "model":  model,
            "RA":     val(d, "RA"),
            f"valid_{model}": bool(d.get("valid", True)),
            # "file":   f,
            "lc_file": re.sub(r"_\d+cs_(linmix|bpl)fit\.pkl$", "", os.path.basename(f)),
        }
        row.update(extract(d, model))
        rows.append(row); n_ok += 1
    frames[model] = pd.DataFrame(rows)
    print(f"[{model}] loaded {n_ok} fits and {len(rows)} rows from {len(files)} files")

if all(len(v) == 0 for v in frames.values()):
    raise SystemExit("no fits loaded; check PAT_SPL / PAT_BPL")
 
# df = pd.DataFrame(rows)
# print(f"total {len(df)} fits ({(df['model']=='spl').sum()} spl, {(df['model']=='bpl').sum()} bpl)")
# os.makedirs(os.path.dirname(OUT_MERGED), exist_ok=True)
# df.to_parquet(OUT_MERGED, index=False)     # pre-crossmatch, so merge isn't lost if catalogues fail
# print("wrote", OUT_MERGED)

# --- 2. widen: identity table + one left join per model ---------------------
# identity first, so a source fitted with only one model is still present
df = (pd.concat([v[ID] for v in frames.values() if len(v)], ignore_index=True)
        .drop_duplicates("lc_file")
        .reset_index(drop=True))
for model, fit in frames.items():
    if not len(fit):
        continue
    # join on lc_file only: RA is a float read independently from each pkl
    fit = fit.drop(columns=[c for c in ID if c != "lc_file"])
    df = df.merge(fit, on="lc_file", how="left", validate="one_to_one")
 
print(f"{len(df)} light curves; "
      f"{df.get('gamma_spl', pd.Series(dtype=float)).notna().sum()} with spl, "
      f"{df.get('gamma_bpl', pd.Series(dtype=float)).notna().sum()} with bpl")
print("fit columns:", [c for c in df.columns if c.endswith(("_spl", "_bpl"))])
 
os.makedirs(os.path.dirname(OUT_MERGED), exist_ok=True)
df.to_parquet(OUT_MERGED, index=False)   # checkpoint, so the pkl loop isn't lost
print("wrote", OUT_MERGED)

# --- 3. crossmatch AGN properties -------------------------------------------
data      = pd.read_csv(CATALOG)
table_mbh = pd.read_csv(MBH, sep=";").rename(columns={"14-150 Lum": "lx"})
_need = ["BAT_ID", "Best_M_BH", "Edd_rat", "L_bol", "lx", "zbest", "NH", "Type_105"]
_missing = [c for c in _need if c not in table_mbh.columns]
if _missing:
    raise SystemExit(f"MBH file {MBH} missing {_missing}; has {list(table_mbh.columns)}")

# --- ZMAD metrics -------------------------------------------------------
zmad = load_zmad()                       # `zmad`, not `z`: AGNPropdf adds a column named z
df = df.merge(zmad, on="lc_file", how="left", validate="one_to_one")
print(f"zmad: matched {df['zmad_ok'].notna().sum()}/{len(df)} light curves "
      f"({(df['zmad_ok'] == True).sum()} ok, {(df['zmad_ok'] == False).sum()} failed, "
      f"{df['zmad_ok'].isna().sum()} no zmad row)")
if (df["zmad_ok"] == False).any():
    print(df.loc[df["zmad_ok"] == False, "zmad_fail_reason"].value_counts())
# df = AGNProp.LC2VarFeatdf(df, data)
# df = AGNProp.AGNPropdf(df, table_mbh)

# silent all-NaN clasf almost always = bat_index dtype mismatch (object vs int)
# print(f"rows={len(df)}  matched_bat_index={df['bat_index'].notna().sum()}  "
#       f"has_clasf={df['clasf'].notna().sum()}")
# print(df["clasf"].value_counts(dropna=False))   # <- use these labels to set TP1/SY/TP2

# df.to_parquet(OUT_PROPS, index=False)
# print("wrote", OUT_PROPS)

# --- 4. BASS properties (once, at the end) ---------------------------------
df = AGNProp.LC2VarFeatdf(df, data)
df = AGNProp.AGNPropdf(df, table_mbh)
 
# all-NaN clasf almost always = bat_index dtype mismatch (object vs int) -> merge misses
print(f"rows={len(df)}  matched_bat_index={df['bat_index'].notna().sum()}  "
      f"has_clasf={df['clasf'].notna().sum()}")
print(df["clasf"].value_counts(dropna=False))   # <- use these labels to set TP1/SY/TP2
 
df.to_parquet(OUT_PROPS, index=False)
print("wrote", OUT_PROPS)
 

# # --- 4.1 plot: one band, coloured by classification --------------------------
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

# --- 5. plot: one band + one model, coloured by classification --------------
# amp  = AMP[MODEL]
# gcol, ycol = f"gamma_{MODEL}", f"{amp}_{MODEL}"
# xerr_cols  = (f"gamma_minerr_{MODEL}", f"gamma_maxerr_{MODEL}")
# yerr_cols  = (f"{amp}_minerr_{MODEL}", f"{amp}_maxerr_{MODEL}")
 
# _missing = [c for c in (gcol, ycol, *xerr_cols, *yerr_cols) if c not in df.columns]
# if _missing:
#     raise SystemExit(f"plot columns absent: {_missing}\n"
#                      f"available: {[c for c in df.columns if c.endswith(('_spl', '_bpl'))]}")
 
# sub = df[(df["band"] == BAND) & (df[f"valid_{MODEL}"] == True) &
#          np.isfinite(df[gcol]) & np.isfinite(df[ycol])]
# print(f"plotting {MODEL} band z{BAND}: {len(sub)} sources")
 
# fig, ax = plt.subplots(figsize=(7, 5))
# for types, color, label in [(TP1, "black", "Type 1"),
#                             (SY,  "C2",    "Sy1.5"),
#                             (TP2, "C1",    "Type 2")]:
#     agn = sub[sub["clasf"].isin(types)]
#     if len(agn) == 0:
#         continue
#     xerr = np.nan_to_num(np.abs(np.vstack([agn[c] for c in xerr_cols])))
#     yerr = np.nan_to_num(np.abs(np.vstack([agn[c] for c in yerr_cols])))
#     kw = dict(fmt="o", alpha=0.3, color=color, label=f"{label} (n={len(agn)})",
#               xerr=xerr, yerr=yerr, elinewidth=0.6, capsize=0)
#     if color == "black":
#         kw.update(markerfacecolor="none", ecolor="k")
#     ax.errorbar(agn[gcol], agn[ycol], **kw)
 
# ax.set_xlabel(r"$\gamma$  (SF slope)")
# ax.set_ylabel(rf"${amp}$")
# ax.tick_params(axis="both", labelsize=12)
# ax.set_xscale("log"); ax.set_yscale("log")
# ax.set_xlim(5e-3, 1e0); ax.set_ylim(5e-3, 1e0)
# ax.grid(True, alpha=0.3)
# ax.set_title(f"z{BAND}  ({MODEL})")
# ax.legend()
# fig.tight_layout()
# fig.savefig(OUT_PNG, dpi=150)
# print("wrote", OUT_PNG)