#!/usr/bin/env python3
"""
plot_fit.py — plot SF data + fitted model for linmix (SPL) or bpl (BPL) fits,
saving each PNG next to its source pkl.

Usage (same style as plot_bplfits.py — a substring/RA filter, plus model & band):
    python plot_fit.py <model> [RA_pattern] [band]
      model      : spl | bpl   (aliases: linmix->spl, single->spl, broken->bpl)
      RA_pattern : optional substring of the path, e.g. an RA like 0.61008
      band       : optional g | r | i  (order-independent: it's the single-letter
                   token, so you can give band with or without an RA)

Examples:
    python plot_fit.py spl                 # all linmix fits
    python plot_fit.py bpl 0.61008         # bpl fits for that RA, all bands
    python plot_fit.py bpl 0.61008 g       # bpl fits for that RA, g band only
    python plot_fit.py spl g               # all linmix fits, g band only
    python plot_fit.py linmix 0.61008      # 'linmix' == 'spl'

Notes:
  * linmix pkls are self-contained (SF data + fit keys inside SF_dict[0]).
  * bpl pkls hold only fit params, so we pair each with its sibling
    <stem>_{X}cs.pkl input for the SF curve.
  * reuses newSF.plot_SF for the drawing; the PNG is moved to the pkl's own
    partials/{ra}_{dec}/ directory, named after the pkl (unique per fit).
"""
import os
import re
import sys
import glob
import shutil
import pickle
from newSF import plot_SF          # newSF sets matplotlib Agg on import

X = 10

# model arg -> (glob for the fit pkls, plot_SF model string)
SPEC = {
    "spl": (f"*_{X}cs_linmixfit.pkl", "single"),
    "bpl": (f"*_{X}cs_bpl_*fit.pkl",  "broken"),
}
ALIASES = {"linmix": "spl", "single": "spl", "broken": "bpl"}


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


def _make_dict(fit_pkl, model_key):
    """Return a dict carrying SF data + fit keys, ready for plot_SF."""
    if model_key == "spl":
        return dict(_unwrap(_load(fit_pkl)))          # SF data + _spl keys inside
    # bpl: params only -> pair with the SF-data input
    fit = _unwrap(_load(fit_pkl))
    src = re.sub(r"_bpl_\w+fit\.pkl$", ".pkl", fit_pkl)
    if not os.path.exists(src):
        raise FileNotFoundError(f"no input SF pkl for bpl fit: {os.path.basename(src)}")
    d = dict(_unwrap(_load(src)))                     # SF/SFmaxerr/SFminerr/band
    d.update(fit)                                     # + A_1_bpl/gamma_bpl/dt_break_bpl
    return d


def plot_one(fit_pkl, model_key, plotsf_model):
    d = _make_dict(fit_pkl, model_key)
    band = d.get("band") or re.search(r"z([gri])_", os.path.basename(fit_pkl)).group(1)

    plot_SF(d, band, plotsf_model)                   # writes plot_SF_<band>_<model>.png in cwd

    tmp  = f"plot_SF_{band}_{plotsf_model}.png"
    dest = fit_pkl[:-len(".pkl")] + ".png"           # same dir as the pkl, unique name
    shutil.move(tmp, dest)
    print(f"  saved {dest}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        sys.exit("usage: python plot_fit.py <spl|bpl> [RA_pattern] [band]")

    model_key = ALIASES.get(args[0].lower(), args[0].lower())
    if model_key not in SPEC:
        sys.exit(f"model must be one of {sorted(set(SPEC) | set(ALIASES))}, got {args[0]!r}")

    rest    = args[1:]
    band    = next((a for a in rest if a in ("g", "r", "i")), None)   # single-letter token
    pattern = next((a for a in rest if a not in ("g", "r", "i")), "")  # the rest is the RA/path

    fit_glob, plotsf_model = SPEC[model_key]
    FIT_GLOB = os.environ["HOME"] + f"/results/partials/*/{fit_glob}"

    def _match(f):
        base = os.path.basename(f)
        return pattern in f and (band is None or re.search(rf"z{band}_", base))

    fits = sorted(f for f in glob.glob(FIT_GLOB) if _match(f))
    print(f"{len(fits)} {model_key} fits found"
          + (f" matching '{pattern}'" if pattern else "")
          + (f" band z{band}" if band else ""))
    for f in fits:
        try:
            plot_one(f, model_key, plotsf_model)
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}")


if __name__ == "__main__":
    main()