"""
check_stage4.py — validate the stage-4 merged files and prove they are
self-sufficient: SF reconstructable, plottable, and re-fittable from the
merged parquets ALONE (no rank files, no LCs).

Usage (serial, compute node, conda env):
    python check_stage4.py                          # format checks, all merged sources
    python check_stage4.py <ra> <dec> <band>        # one source, verbose
    python check_stage4.py --plot <ra> <dec> <band> [oi ...]
        # SF + cached power-law overlay from merged files (no MCMC).
        # default sims: 0 1 2. PNGs -> ~/results/stage4/plots/
    python check_stage4.py --refit <ra> <dec> <band> <oi> <cadence>
        # full proof: reconstruct SF from merged parquet, re-run the seeded
        # LinMix fit, compare all FIT_KEYS to the merged fit row.
        # (Chains byte-match only for fits produced with parallelize=False.)

Format checks
─────────────
fit_{tag}.parquet : exact columns, dtypes, 240 unique (oi, cadence) rows,
                    valid rows have finite FIT_KEYS
sf_{tag}.parquet  : exact columns, all strings, no '...' elision;
                    sampled rows round-trip through parse_to_dict and pass the
                    SF_linmix preprocessing dry-run; dry-run outcome is
                    cross-checked against the merged fit's valid flag
"""

import os
import re
import sys
import glob
import hashlib
import numpy as np
import pandas as pd

from newSF import parse_to_dict

S4       = os.environ["HOME"] + "/results/stage4/"
PLOTS    = S4 + "plots/"
N_SIMS   = 120
CADENCES = ("full", "crop")
mode = 'spl'
FIT_KEYS = [
   f"A_1_{mode}", f"A_365_{mode}", f"A_maxerr_{mode}", f"A_minerr_{mode}",
    f"gamma_{mode}", f"gamma_maxerr_{mode}", f"gamma_minerr_{mode}",
]
FIT_WINDOW = (1.0, 365.0)
N_SAMPLE   = 8


def linmix_seed(ra, dec, band, oi, cadence):
    """Must match stage3_sf_linmix.py exactly."""
    s = f"{ra}_{dec}_{band}_{oi}_{cadence}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16) & 0x7FFFFFFF


def find_sources():
    out = []
    for p in glob.glob(S4 + "fit_*.parquet"):
        m = re.search(r"fit_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.parquet",
                      os.path.basename(p))
        if m:
            out.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return sorted(out)


def load(ra, dec, band):
    tag = f"{ra}_{dec}_z{band}"
    fp, sp = S4 + f"fit_{tag}.parquet", S4 + f"sf_{tag}.parquet"
    fit = pd.read_parquet(fp) if os.path.exists(fp) else None
    sf  = pd.read_parquet(sp) if os.path.exists(sp) else None
    return fit, sf


def sf_linmix_dry_run(d):
    try:
        ii = pd.IntervalIndex(d["SF"].index)
    except Exception as e:
        return False, f"IntervalIndex failed: {e}"
    win = (ii.left >= 1) & (ii.right <= 365) & (d["SF"] != 0)
    sf_mag = d["SF"][win].dropna()
    if len(sf_mag) <= 3:
        return False, f"too few points ({len(sf_mag)})"
    maxe = d["SFmaxerr"].reindex(sf_mag.index)
    mine = d["SFminerr"].reindex(sf_mag.index)
    if maxe.isna().any() or mine.isna().any():
        return False, "reindex NaN — interval mismatch"
    if (((maxe + mine) / 2) <= 0).any():
        return False, "non-positive errors"
    return True, ""


def parse_row(srow):
    return {k: parse_to_dict(srow[k])["SF"] for k in ("SF", "SFmaxerr", "SFminerr")}


# ── format checks ─────────────────────────────────────────────────────────────

def check_source(ra, dec, band, verbose=False):
    tag = f"{ra}_{dec}_z{band}"
    fit, sf = load(ra, dec, band)
    print(f"{tag}:")
    ok = True

    if fit is None:
        print("  [FAIL] no merged fit parquet")
        return False

    need_fit = {"object_index", "cadence", "valid", "fail_reason", *FIT_KEYS}
    if set(fit.columns) != need_fit:
        print(f"  [FAIL] fit columns: missing={need_fit-set(fit.columns)} "
              f"extra={set(fit.columns)-need_fit}")
        ok = False

    pairs = list(zip(fit["object_index"], fit["cadence"]))
    expected = {(oi, c) for oi in range(N_SIMS) for c in CADENCES}
    if set(pairs) != expected or len(pairs) != len(expected):
        print(f"  [FAIL] fit rows {len(pairs)}, unique {len(set(pairs))}, "
              f"expected {len(expected)}")
        ok = False

    v = fit[fit["valid"]]
    for k in FIT_KEYS:
        if len(v) and not np.isfinite(v[k]).all():
            print(f"  [FAIL] non-finite {k} in valid rows")
            ok = False

    if sf is None:
        print("  [warn] no merged SF parquet (run stage4_merge.py --sf) — "
              "SF checks skipped")
    else:
        need_sf = {"object_index", "cadence", "SF", "SFmaxerr", "SFminerr"}
        if set(sf.columns) != need_sf:
            print(f"  [FAIL] sf columns {sorted(sf.columns)}")
            ok = False
        for col in ("SF", "SFmaxerr", "SFminerr"):
            if (~sf[col].apply(lambda x: isinstance(x, str))).sum():
                print(f"  [FAIL] non-string entries in {col}")
                ok = False
            if sf[col].str.contains(r"\.\.\.", regex=True).sum():
                print(f"  [FAIL] elided ('...') strings in {col}")
                ok = False

        # sampled round-trip + dry-run vs fit validity
        sample = list(sf.index[:N_SAMPLE])
        if len(v):
            m = sf[(sf.object_index == v.iloc[0]["object_index"])
                   & (sf.cadence == v.iloc[0]["cadence"])]
            sample += list(m.index[:1])
        for i in sorted(set(sample)):
            srow = sf.loc[i]
            d = parse_row(srow)
            if not all(isinstance(x, pd.Series) for x in d.values()):
                print(f"  [FAIL] (oi={srow['object_index']},{srow['cadence']}) "
                      f"parse_to_dict failure")
                ok = False
                continue
            would, why = sf_linmix_dry_run(d)
            frow = fit[(fit.object_index == srow["object_index"])
                       & (fit.cadence == srow["cadence"])]
            if len(frow) and bool(frow.iloc[0]["valid"]) and not would:
                print(f"  [FAIL] (oi={srow['object_index']},{srow['cadence']}) "
                      f"valid fit but reconstructed SF unfittable: {why}")
                ok = False
            elif verbose:
                print(f"    (oi={srow['object_index']},{srow['cadence']}): "
                      f"round-trip ok, dry-run {'fit' if would else 'skip'}")

    if verbose and len(v):
        for cad in CADENCES:
            g = v.loc[v["cadence"] == cad, "_gamma_spl"]
            if len(g):
                print(f"  gamma[{cad}]: median {g.median():.3f} "
                      f"[{g.quantile(.16):.3f}, {g.quantile(.84):.3f}] n={len(g)}")
    print(f"  => {'[ OK ]' if ok else '[FAIL]'}")
    return ok


# ── plot from merged files only ───────────────────────────────────────────────

def plot(ra, dec, band, ois):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOTS, exist_ok=True)
    tag = f"{ra}_{dec}_z{band}"
    fit, sf = load(ra, dec, band)
    print(fit)
    print(sf)
    if fit is None or sf is None:
        print(f"{tag}: need BOTH merged fit and SF parquets (--sf merge)")
        return

    fig, axes = plt.subplots(len(ois), 2, figsize=(11, 3.2 * len(ois)),
                             squeeze=False)
    for r, oi in enumerate(ois):
        for c, cad in enumerate(CADENCES):
            ax = axes[r][c]
            ax.set_title(f"sim {oi} — {cad}", fontsize=9)
            srow = sf[(sf.object_index == oi) & (sf.cadence == cad)]
            frow = fit[(fit.object_index == oi) & (fit.cadence == cad)]
            if len(srow) == 0:
                ax.text(.5, .5, "no SF", ha="center", transform=ax.transAxes)
                continue
            d = parse_row(srow.iloc[0])
            s = d["SF"].dropna()
            cats = s.index.categories[s.index.codes]
            mid, half = cats.mid.values, (cats.length / 2).values
            maxe = d["SFmaxerr"].reindex(s.index).values
            mine = d["SFminerr"].reindex(s.index).values
            color = "#436BAD" if cad == "full" else "orange"
            ax.errorbar(mid, s.values, xerr=half, yerr=(mine, maxe),
                        fmt="o", ms=4, capsize=2, c=color, label=f"SF ({cad})")
            if len(frow) and bool(frow.iloc[0]["valid"]):
                A365 = frow.iloc[0]["_A_365_spl"]
                gam  = frow.iloc[0]["_gamma_spl"]
                dtl = np.logspace(0, np.log10(365), 100)
                ax.plot(dtl, A365 * (dtl / 365.0) ** gam, "-", lw=2, c="crimson",
                        label=f"fit: γ={gam:.2f}")
                ax.axvspan(*FIT_WINDOW, alpha=.06, color="grey")
            elif len(frow):
                ax.set_title(f"sim {oi} — {cad} [INVALID: "
                             f"{frow.iloc[0]['fail_reason']}]",
                             fontsize=8, color="crimson")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Δt [days]"); ax.set_ylabel("SF [mag]")
            ax.legend(fontsize=7)
    fig.suptitle(f"{tag}: merged-parquet SF + cached fit")
    fig.tight_layout()
    out = PLOTS + f"s4_{tag}_sims{'-'.join(map(str, ois))}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"plot -> {out}")


# ── refit proof from merged files only ────────────────────────────────────────

def refit(ra, dec, band, oi, cadence):
    from newSF import SF_linmix

    tag = f"{ra}_{dec}_z{band}"
    fit, sf = load(ra, dec, band)
    if fit is None or sf is None:
        print(f"{tag}: need both merged parquets")
        return False
    srow = sf[(sf.object_index == oi) & (sf.cadence == cadence)]
    frow = fit[(fit.object_index == oi) & (fit.cadence == cadence)]
    if len(srow) == 0 or len(frow) == 0:
        print(f"{tag} sim {oi} {cadence}: not in merged files")
        return False
    frow = frow.iloc[0]
    if not bool(frow["valid"]):
        print(f"{tag} sim {oi} {cadence}: cached fit invalid "
              f"({frow['fail_reason']}) — pick a valid one")
        return False

    d = parse_row(srow.iloc[0])
    np.random.seed(linmix_seed(ra, dec, band, oi, cadence))
    res = SF_linmix(d)
    if res is None:
        print("[FAIL] refit skipped/failed on reconstructed SF")
        return False

    print(f"\n{tag} sim {oi} {cadence} — merged-cache vs refit:")
    all_match = True
    for k in FIT_KEYS:
        a, b = float(frow[k]), float(res[k])
        m = np.isclose(a, b, rtol=1e-9, atol=1e-12)
        all_match &= m
        print(f"  {k:22s} cached {a:+.9e}  refit {b:+.9e}"
              f"{'' if m else '   <-- MISMATCH'}")
    print("[ OK ] merged files are refit-complete" if all_match else
          "[FAIL] mismatch — version/parallelize drift or round-trip issue")
    return all_match


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--plot":
        ra, dec, band = float(a[1]), float(a[2]), a[3]
        ois = [int(x) for x in a[4:]] or [0, 1, 2]
        plot(ra, dec, band, ois)
    elif a and a[0] == "--refit":
        ra, dec, band, oi, cad = float(a[1]), float(a[2]), a[3], int(a[4]), a[5]
        refit(ra, dec, band, oi, cad)
    elif len(a) == 3:
        check_source(float(a[0]), float(a[1]), a[2], verbose=True)
    else:
        srcs = find_sources()
        print(f"found {len(srcs)} merged sources\n")
        n = sum(check_source(*s) for s in srcs)
        print(f"\n{n}/{len(srcs)} sources passed")
        sys.exit(0 if n == len(srcs) and len(srcs) > 0 else 1)