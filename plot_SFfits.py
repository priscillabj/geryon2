#!/usr/bin/env python3
"""
plot_SFfits.py — plot SF data + fitted model for linmix (SPL) or bpl (BPL) fits,
saving each PNG next to its source pkl.

Two mutually exclusive selection modes:

  1. glob mode (original):
         python plot_SFfits.py <model> [RA_pattern] [band]
           model      : spl | bpl   (aliases: linmix->spl, single->spl, broken->bpl)
           RA_pattern : optional substring of the path, e.g. an RA like 0.61008
           band       : optional g | r | i  (order-independent single-letter token)

  2. list mode (for driving off a DataFrame subsample):
         python plot_SFfits.py --p-list FILE|-
       FILE is a JSON array, either of fit-pkl paths:
           ["/.../foo_10cs_linmixfit.pkl", ...]
       or of objects carrying the parent light curve as well (needed by
       --diagnostic):
           [{"pkl": "/.../foo_10cs_linmixfit.pkl", "lc": "/.../foo_merged.parquet"}]
       The model (spl/bpl) is inferred per path from the filename, so a mixed
       list is fine. Input order is preserved.

Both modes accept --limit / --offset (repo convention: offset applied first).

Examples:
    python plot_SFfits.py spl                     # all linmix fits
    python plot_SFfits.py bpl 0.61008 g           # bpl fits, that RA, g band
    python plot_SFfits.py --p-list pkls.json --limit 20
    pandas ... | python plot_SFfits.py --pkl-list - --diagnostic

Notes:
  * linmix pkls are self-contained (SF data + fit keys inside SF_dict[0]).
  * bpl pkls hold only fit params, so we pair each with its sibling
    <stem>_{X}cs.pkl input for the SF curve.  Current bpl outputs are named
    <stem>_{X}cs{flat|at_break}_bplfit.pkl; the legacy <stem>_{X}cs_bpl_*fit.pkl
    form is still recognised.
  * reuses newSF.plot_SF for the drawing; the PNG is moved to the pkl's own
    partials/{ra}_{dec}/ directory, named after the pkl (unique per fit, so the
    flat and at_break variants do not collide).
  * --diagnostic additionally re-runs newSF.optSF(lc, save_plt=True), which is
    the only way to get the newSF.plot_sf two-panel LC+SF figure: that function
    takes optSF internals that are not stored in the pkl.  This RECOMPUTES the
    SF and is orders of magnitude slower than plotting; pass the same kwargs the
    original run used or the panel will not match the fitted curve.
"""
import os
import re
import sys
import json
import glob
import shutil
import pickle
import argparse
from newSF import plot_SF, optSF   # newSF sets matplotlib Agg on import

X = 10

# model key -> (globs for the fit pkls, plot_SF model string)
SPEC = {
    "spl": ((f"*_{X}cs_linmixfit.pkl",), "single"),
    "bpl": ((f"*_{X}cs*_bplfit.pkl",            # current: ..._10csflat_bplfit.pkl
             f"*_{X}cs_bpl_*fit.pkl"), "broken"),  # legacy:  ..._10cs_bpl_flatfit.pkl
}
ALIASES = {"linmix": "spl", "single": "spl", "broken": "bpl"}

# strips either bpl suffix back to the input SF pkl
BPL_SUFFIX = re.compile(r"(?:(?:flat|at_break)_bplfit|_bpl_\w+fit)\.pkl$")


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _unwrap(d):
    """drivers / SF_wnoise may wrap the dict as {'SF_dict':[d]} or [d]."""
    if isinstance(d, dict) and "SF_dict" in d:
        return d["SF_dict"][0]
    if isinstance(d, list):
        return d[0]
    return d


def _model_of(path):
    """spl | bpl inferred from the fit-pkl filename, else None."""
    base = os.path.basename(path)
    if base.endswith("_linmixfit.pkl"):
        return "spl"
    if BPL_SUFFIX.search(base):
        return "bpl"
    return None


def _make_dict(fit_pkl, model_key):
    """Return a dict carrying SF data + fit keys, ready for plot_SF."""
    if model_key == "spl":
        return dict(_unwrap(_load(fit_pkl)))          # SF data + _spl keys inside
    # bpl: params only -> pair with the SF-data input
    fit = _unwrap(_load(fit_pkl))
    src = BPL_SUFFIX.sub(".pkl", fit_pkl)
    if src == fit_pkl:
        raise ValueError(f"unrecognised bpl suffix: {os.path.basename(fit_pkl)}")
    if not os.path.exists(src):
        raise FileNotFoundError(f"no input SF pkl for bpl fit: {os.path.basename(src)}")
    d = dict(_unwrap(_load(src)))                     # SF/SFmaxerr/SFminerr/band
    d.update(fit)                                     # + A_1_bpl/gamma_bpl/dt_break_bpl
    return d


def plot_one(fit_pkl, model_key, plotsf_model):
    d = _make_dict(fit_pkl, model_key)
    band = d.get("band") or re.search(r"z([gri])_", os.path.basename(fit_pkl)).group(1)

    plot_SF(d, band, plotsf_model)                    # writes plot_SF_<band>_<model>.png in cwd

    tmp  = f"plot_SF_{band}_{plotsf_model}.png"
    dest = fit_pkl[:-len(".pkl")] + ".png"            # same dir as the pkl, unique name
    shutil.move(tmp, dest)
    print(f"  saved {dest}")


def _jobs_from_list(src):
    """JSON array of paths or {'pkl','lc'} objects -> [(fit_pkl, model_key, lc)]."""
    stream = sys.stdin if src == "-" else open(src)
    seen, jobs = set(), []
    for entry in json.load(stream):
        if isinstance(entry, str):
            fit_pkl, lc = entry, None
        else:
            fit_pkl, lc = entry["pkl"], entry.get("lc")
        if fit_pkl in seen:
            continue
        seen.add(fit_pkl)
        model_key = _model_of(fit_pkl)
        if model_key is None:
            print(f"  [skip] {os.path.basename(fit_pkl)}: not a recognised fit pkl")
        else:
            jobs.append((fit_pkl, model_key, lc))
    print(f"{len(jobs)} fits from {src}")
    return jobs


def _jobs_from_glob(model_key, rest):
    band    = next((a for a in rest if a in ("g", "r", "i")), None)
    pattern = next((a for a in rest if a not in ("g", "r", "i")), "")
    root    = os.environ["HOME"] + "/results/partials/*/"

    def _match(f):
        base = os.path.basename(f)
        return pattern in f and (band is None or re.search(rf"z{band}_", base))

    fits = set()
    for g in SPEC[model_key][0]:
        fits.update(f for f in glob.glob(root + g) if _match(f))
    fits = sorted(fits)
    print(f"{len(fits)} {model_key} fits found"
          + (f" matching '{pattern}'" if pattern else "")
          + (f" band z{band}" if band else ""))
    return [(f, model_key, None) for f in fits]


def main():
    p = argparse.ArgumentParser(
        description="plot SF data + fitted model for spl (linmix) or bpl fits")
    p.add_argument("model", nargs="?",
                   help="spl | bpl (aliases: linmix, single, broken)")
    p.add_argument("rest", nargs="*",
                   help="[RA_pattern] [band] — order-independent")
    p.add_argument("--p-list", metavar="FILE",
                   help="JSON array of fit pkl paths, or of {'pkl','lc'} objects "
                        "('-' for stdin); mutually exclusive with the positionals")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--diagnostic", action="store_true",
                   help="also re-run optSF(lc, save_plt=True) for the newSF.plot_sf "
                        "LC+SF panel; recomputes the SF, needs 'lc' in --pkl-list")
    args = p.parse_args()

    if args.p_list and args.model:
        p.error("--p-list takes no positional args")
    if not args.p_list and not args.model:
        p.error("give <model> [RA_pattern] [band], or --p-list FILE")

    if args.p_list:
        jobs = _jobs_from_list(args.p_list)
    else:
        model_key = ALIASES.get(args.model.lower(), args.model.lower())
        if model_key not in SPEC:
            p.error(f"model must be one of {sorted(set(SPEC) | set(ALIASES))}, "
                    f"got {args.model!r}")
        jobs = _jobs_from_glob(model_key, args.rest)

    if args.limit or args.offset:
        jobs = jobs[args.offset:args.offset + args.limit if args.limit else None]
        print(f"  -> {len(jobs)} after offset={args.offset} limit={args.limit}")

    for fit_pkl, model_key, lc in jobs:
        try:
            plot_one(fit_pkl, model_key, SPEC[model_key][1])
        except Exception as e:
            print(f"  [skip] {os.path.basename(fit_pkl)}: {e}")
            continue
        if args.diagnostic:
            if lc is None:
                print(f"  [skip diag] {os.path.basename(fit_pkl)}: no 'lc' path")
                continue
            try:
                optSF(lc, save_plt=True, x=X)
            except Exception as e:
                print(f"  [skip diag] {os.path.basename(lc)}: {e}")


if __name__ == "__main__":
    main()