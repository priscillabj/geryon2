#!/usr/bin/env python
"""
Build one master table per light curve: SF fits (both models) + ZMAD noise
metrics + BASS AGN properties, then optionally plot amplitude vs gamma.

  python -u merge_AGNprop.py --list ~/results/input_files.json
  python -u merge_AGNprop.py --list ~/results/input_files.json --limit 20 --plots

File selection mirrors runDRW_SF.select_files: --list is a JSON array of
light-curve parquet names, which may mix merged and single-field forms

    0.20323_-7.15322_zg_merged.parquet              (merged)
    0.20323_-7.15322_000395_zi_ccd13_q3.parquet     (single field/ccd/quadrant)

and may span bands, so --band is an optional filter (default: all bands).

Output columns:
  <param>[_minerr|_maxerr]_<model>   gamma_spl, A_365_maxerr_spl, A_break_bpl, ...
  zmad_<metric>                      zmad_sigma, zmad_ok, zmad_fail_reason, ...
  valid_<model>, pkl_<model>         per-model fit flag and provenance
  merged                             True for *_merged.parquet light curves
  sci                                True for *_zX_sci_merged.parquet variants

NOTE 1: A_365_spl and A_break_bpl are different quantities -- don't pair them.
NOTE 2: merged and single-field light curves have different baselines, epoch
counts and zero-point treatment, so `band` alone does not make them comparable;
filter on `merged` before any cross-band comparison.
NOTE 3: (source, band) is NOT unique -- a source can have several single-field
files in one band, and a *_sci_merged variant alongside its *_merged twin. Only
lc_file is unique. Without --list the partials glob picks up sci variants too;
with --list they are excluded unless the JSON names them.

Serial, seconds. AGNProp.py and sf_keys.py must be importable.
"""
import argparse, glob, json, os, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import AGNProp
import sf_keys                       # single source of truth for fit-key names

PARTIALS_ROOT = Path(os.environ["HOME"]) / "results" / "partials"
# OUT_MERGED = os.path.expanduser("~/results/SFfit_merged_zmadflux.parquet")
OUT_MERGED = os.path.expanduser("~/results/SFfit_tmp.parquet")
# OUT_PROPS  = os.path.expanduser("~/results/bat_master_zmadflux_sigma>10.parquet")
# OUT_PNG    = os.path.expanduser("~/results/amp_vs_gamma_byclasf_sigma>10_2.png")
OUT_PROPS  = os.path.expanduser("~/results/bat_master.parquet")
OUT_PNG    = os.path.expanduser("~/results/amp_vs_gamma_byclasf_all2.png")

# --- catalogue paths on geryon2 -------------------------------------------
# data:      wise_ra, wise_dec, AGN ('TRUE'/'FALSE' strings), bat_index, ctpt_name
CATALOG = os.path.expanduser("~/DR2-105monthcatalog-105m_all.csv")
MBH     = os.path.expanduser("~/DR2_DR3_Best MBH - DR2_final_use_this.csv")

# --- ZMAD -----------------------------------------------------------------
ZMAD_PAT      = os.path.expanduser("~/results/zmad_all_flux.parquet")
ZMAD_APERTURE = "4"
ZMAD_METRICS  = ["ok", "fail_reason", "sigma", "perc", "agn", "cs_mean", "cs_std",
                 "cs_median", "cs_max", "n_stars", "n_stars_preclip", "n_epochs", "space"]

# --- plot config (only used with --plots) ---------------------------------
BAND  = "g"
MODEL = "spl"
AMP   = {"spl": "A_365", "bpl": "A_break"}
# reconcile with the printed clasf.value_counts() before trusting the plot:
TP1 = ["Sy1", "Sy1.2"]             # Type 1        -> black, hollow markers
SY  = ["Sy1.5"]                    # intermediate  -> C2
TP2 = ["Sy1.8", "Sy1.9", "Sy2"]    # Type 2        -> C1
# --------------------------------------------------------------------------

# FIT_TAIL = {"spl": "linmixfit", "bpl": "bplfit"}
FIT_TAIL = {"spl": "{x}cs_linmixfit", "bpl": "{x}csflat_bplfit"}
band_re  = re.compile(r"_z(\w)_")          # matches _zg_merged AND _zi_ccd13_
# pkl_re   = re.compile(r"_\d+cs_(linmix|bpl)fit\.pkl$")
pkl_re   = re.compile(r"_(?:%s)\.pkl$" % "|".join(
    re.escape(t).replace(r"\{x\}", r"\d+") for t in FIT_TAIL.values()))
_STRIP   = re.compile(r"_(spl|bpl)$")
ID       = ["lc_file", "source", "band", "merged", "sci", "RA", "med_mag"]


def val(d, name):
    """value for `name`, mapping absent/None -> NaN."""
    v = d.get(name)
    return np.nan if v is None else v


def extract(d, model):
    """Canonical fit keys for `model` (defined once in sf_keys), re-tagged with a
    uniform _<model> suffix so nothing collides between models. A no-op for spl
    and bpl (their sf_keys names already carry it); does real work for
    spl_legacy, whose keys are '_A_1', '_A_maxerr_spl', ..."""
    out = {}
    for key in sf_keys.keys_for(model):
        base = _STRIP.sub("", key).lstrip("_")
        out[f"{base}_{model}"] = val(d, key)
    return out


def pkl_pattern(model, x):
    return str(PARTIALS_ROOT / "*" / f"*_z*.parquet_{FIT_TAIL[model].format(x=x)}.pkl")


def out_paths(args):
    """--limit writes to *.testN.parquet so a truncated test run can never
    overwrite the real master. Delete this function and use the constants
    directly if you'd rather it didn't."""
    if not args.limit:
        return OUT_MERGED, OUT_PROPS, OUT_PNG
    tag = f".test{args.limit}"
    add = lambda p: p.replace(".parquet", f"{tag}.parquet").replace(".png", f"{tag}.png")
    return add(OUT_MERGED), add(OUT_PROPS), add(OUT_PNG)


def wanted_names(args):
    """The set of light-curve basenames to merge.

    Built from --list when given, otherwise from the partials tree itself. Either
    way it is resolved ONCE and shared by both models: applying --limit to each
    model's glob separately would hand spl and bpl different subsets, and the
    wide join would then be mostly half-empty rows."""
    if args.list:
        names = json.loads(Path(os.path.expanduser(args.list)).read_text())
        w = {os.path.basename(n) for n in names}      # tolerate abs paths in the JSON
        src = f"--list {args.list}"
    else:
        w = {pkl_re.sub("", os.path.basename(p))
             for model in FIT_TAIL
             for p in glob.glob(pkl_pattern(model, args.x))}
        src = "partials tree"
    n_all = len(w)

    if args.band:
        w = {n for n in w if re.search(rf"_z{args.band}_", n)}
    if not w:
        raise SystemExit(f"{src}: no entries for band {args.band!r} at --x {args.x}")

    if args.limit:
        w = set(sorted(w)[:args.limit])               # sorted -> reproducible subset
        print(f"*** --limit {args.limit}: TEST RUN, output is not the full master ***")

    print(f"{src}: {n_all} light curves -> {len(w)} selected"
          f"{f' (band {args.band})' if args.band else ''}")
    return w


def select_pkls(model, wanted, args):
    """Glob the partials tree, keep only fits for `wanted`. Filtering on the
    basename before unpickling is what makes this cheap; building the paths from
    the JSON instead would hardcode the partials layout ({ra}_{dec}) and the
    {x}cs count in a second place, and would silently miss any source whose
    directory doesn't follow that convention."""
    paths = sorted(glob.glob(pkl_pattern(model, args.x)))
    keep = [(lc, p) for p in paths
            if (lc := pkl_re.sub("", os.path.basename(p))) in wanted]
    print(f"[{model}] {len(keep)} of {len(paths)} pkls selected")
    missing = wanted - {lc for lc, _ in keep}
    if missing:
        print(f"[{model}] {len(missing)} selected light curves have no {model} fit, "
              f"e.g. {sorted(missing)[:3]}")
    return keep


def load_fits(model, wanted, args):
    """One row per light curve for this model, fit columns suffixed _<model>."""
    rows, printed = [], False
    for lc, f in select_pkls(model, wanted, args):
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception as e:
            print(f"skip (load fail) {f}: {e}"); continue
        if not isinstance(d, dict):
            print(f"skip (not dict) {f}"); continue
        if not printed:                      # verify real key names once per model
            print(f"[{model}] dict keys:", sorted(d.keys())); printed = True
        m = band_re.search(lc)
        rows.append({
            "lc_file": lc,                             # == the --list entry
            "source":  os.path.basename(os.path.dirname(f)),
            "band":    m.group(1) if m else "?",
            "merged":  lc.endswith("_merged.parquet"),
            "sci":     "_sci_" in lc,      # *_zr_sci_merged.parquet: separate reduction
            "RA":      val(d, "RA"),
            "med_mag": val(d, "mag"),
            # "epochs":  val(d, "epochs"),
            f"valid_{model}": bool(d.get("valid", True)),
            f"pkl_{model}":   f,
            **extract(d, model),
        })
    print(f"[{model}] loaded {len(rows)} fits")
    return pd.DataFrame(rows)


def load_zmad(pattern=ZMAD_PAT, aperture=ZMAD_APERTURE):
    """One row per light-curve file, metrics prefixed zmad_, keyed on lc_file.
    Failure rows (aperture is null) are kept so 'ran and failed' stays
    distinguishable from 'never run'. Not filtered by --list/--limit: it
    left-joins onto df, so surplus rows simply don't match."""
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
                         f"-- overlapping files, or mag and flux runs mixed:\n"
                         f"{z.loc[dup, ['lc_file', 'space', 'agg', 'col']].head(10)}")
    cols = [c for c in ZMAD_METRICS if c in z.columns]
    return z[["lc_file"] + cols].rename(columns={c: f"zmad_{c}" for c in cols})


def read_catalogues():
    """Read and validate up front: a second costs nothing, and both the
    semicolon-delimiter and AGN-as-bool failures are otherwise invisible until
    after the pkl loop."""
    data      = pd.read_csv(CATALOG)
    table_mbh = pd.read_csv(MBH, sep=",").rename(columns={"14-150 Lum": "lx"})

    need = ["BAT_ID", "Best_M_BH", "Edd_rat", "L_bol", "lx", "zbest", "NH", "Type_105"]
    missing = [c for c in need if c not in table_mbh.columns]
    if missing:
        raise SystemExit(f"MBH file {MBH} missing {missing}; has {list(table_mbh.columns)}")

    need = ["wise_ra", "wise_dec", "AGN", "bat_index", "ctpt_name"]
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise SystemExit(f"catalogue {CATALOG} missing {missing}; has {list(data.columns)}")
    if data["AGN"].dtype == bool:
        raise SystemExit("data.AGN parsed as bool; LC2VarFeatdf compares == 'TRUE'.")
    return data, table_mbh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", default=None,
                   help="JSON list of light-curve parquet names; only these are merged")
    p.add_argument("--band", default=None,
                   help="optional band filter (g/r/i). Default: all bands -- a "
                        "single-band master is almost never what you want here")
    p.add_argument("--x", type=int, default=10,
                   help="calstar count in the fit pkl name")
    p.add_argument("--limit", type=int, default=None,
                   help="TESTING ONLY: merge just the first N light curves "
                        "(sorted). Output goes to *.testN.parquet")
    p.add_argument("--plots", action="store_true")
    args = p.parse_args()

    out_merged, out_props, out_png = out_paths(args)
    data, table_mbh = read_catalogues()
    wanted = wanted_names(args)

    # --- 1. load both models -----------------------------------------------
    frames = {m: load_fits(m, wanted, args) for m in FIT_TAIL}
    if all(len(v) == 0 for v in frames.values()):
        raise SystemExit("no fits loaded; check --list, --x and PARTIALS_ROOT")

    # --- 2. widen: identity table + one left join per model ----------------
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
    print(df.groupby(["band", "merged", "sci"], dropna=False).size())
    print("fit columns:", [c for c in df.columns if c.endswith(("_spl", "_bpl"))])

    os.makedirs(os.path.dirname(out_merged), exist_ok=True)
    df.to_parquet(out_merged, index=False)   # checkpoint, so the pkl loop isn't lost
    print("wrote", out_merged)

    # --- 3. ZMAD metrics ----------------------------------------------------
    zmad = load_zmad()          # `zmad`, not `z`: AGNPropdf adds a column named z
    df = df.merge(zmad, on="lc_file", how="left", validate="one_to_one")
    print(f"zmad: matched {df['zmad_ok'].notna().sum()}/{len(df)} light curves "
          f"({(df['zmad_ok'] == True).sum()} ok, {(df['zmad_ok'] == False).sum()} failed, "
          f"{df['zmad_ok'].isna().sum()} no zmad row)")
    if (df["zmad_ok"] == False).any():
        print(df.loc[df["zmad_ok"] == False, "zmad_fail_reason"].value_counts())

    # --- 4. BASS properties (once, at the end) ------------------------------
    n0 = len(df)
    df = AGNProp.LC2VarFeatdf(df, data)
    df = AGNProp.AGNPropdf(df, table_mbh)
    if len(df) != n0:
        # print(f"WARNING: crossmatch changed row count {n0} -> {len(df)}")
        raise SystemExit(f"crossmatch changed row count {n0} -> {len(df)}; "
                     f"duplicate or null keys in the catalogues")

    # all-NaN clasf almost always = bat_index dtype mismatch (object vs int)
    print(f"rows={len(df)}  matched_bat_index={df['bat_index'].notna().sum()}  "
          f"has_clasf={df['clasf'].notna().sum()}")
    print(df["clasf"].value_counts(dropna=False))   # <- labels for TP1/SY/TP2

    df.to_parquet(out_props, index=False)
    print("wrote", out_props)

    if args.plots:
        plot(df, out_png)


def plot(df, out_png):
    """One band + one model, coloured by classification."""
    amp = AMP[MODEL]
    gcol, ycol = f"gamma_{MODEL}", f"{amp}_{MODEL}"
    xerr_cols  = (f"gamma_minerr_{MODEL}", f"gamma_maxerr_{MODEL}")
    yerr_cols  = (f"{amp}_minerr_{MODEL}", f"{amp}_maxerr_{MODEL}")

    missing = [c for c in (gcol, ycol, *xerr_cols, *yerr_cols) if c not in df.columns]
    if missing:
        raise SystemExit(f"plot columns absent: {missing}\navailable: "
                         f"{[c for c in df.columns if c.endswith(('_spl', '_bpl'))]}")

    sub = df[(df["band"] == BAND) & (df["merged"] == True) & (df["sci"] == False) &
             (df[f"valid_{MODEL}"] == True) &
             np.isfinite(df[gcol]) & np.isfinite(df[ycol])]
    print(f"plotting {MODEL} band z{BAND} (merged only): {len(sub)} sources")

    fig, ax = plt.subplots(figsize=(7, 5))
    for types, color, label in [(TP1, "black", "Type 1"),
                                (SY,  "#E69F00",    "Sy1.5"),
                                (TP2, "#56B4E9",    "Type 2")]:
        agn = sub[sub["clasf"].isin(types)]
        if len(agn) == 0:
            continue
        xerr = np.nan_to_num(np.abs(np.vstack([agn[c] for c in xerr_cols])))
        yerr = np.nan_to_num(np.abs(np.vstack([agn[c] for c in yerr_cols])))
        kw = dict(fmt="o", alpha=0.3, color=color, label=f"{label} (n={len(agn)})",
                  xerr=xerr, yerr=yerr, elinewidth=0.6, capsize=0)
        if color == "black":
            kw.update(markerfacecolor="none", ecolor="k")
        ax.errorbar(agn[gcol], agn[ycol], **kw)

    ax.set_xlabel(r"$\gamma$  (SF slope)")
    ax.set_ylabel(rf"${amp}$")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(5e-3, 1e0); ax.set_ylim(5e-3, 1e0)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"z{BAND}  ({MODEL})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


if __name__ == "__main__":
    main()