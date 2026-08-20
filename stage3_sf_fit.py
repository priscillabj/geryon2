"""
stage3_sf_fit.py — stage 3 with selectable SF model.

    mpirun -n <nranks> python stage3_sf_fit.py --spl     # SF_linmix   (single power law)
    mpirun -n <nranks> python stage3_sf_fit.py --linmix  # alias of --spl
    mpirun -n <nranks> python stage3_sf_fit.py --bpl     # bpl_mcmc    (broken power law)

A model MUST be given explicitly — no default, so a mis-typed flag can never
silently fit the wrong model into the wrong cache.

Cache layout (SPL paths are byte-identical to stage3_sf_linmix.py, so the
existing 1000+ sources of SPL work is preserved and skipped on resume):
    SPL  fits : fit_{tag}_rank{r}.jsonl        sentinel sim_{tag}.stage3.done
    BPL  fits : fitbpl_{tag}_rank{r}.jsonl     sentinel sim_{tag}.stage3bpl.done
    SF (both) : sf_{tag}_rank{r}.parquet       <-- SHARED, model-independent

Cadences are DISCOVERED per source from CADENCE_FILES: any cadence whose LC
file is absent is skipped for that source (fewer rows, no error), and a
cadence already present in the SF cache can be fitted without its LC at all.

SF reuse: the SF is model-independent, so if it is already cached (e.g. from
the SPL run) this script reads it back via parse_to_dict and NEVER recomputes
SF or touches the stage-1/2 light curves. Running --bpl over sources already
processed with --spl therefore costs only the BPL fits. All rank SF files for
a source are pooled, so reuse works even at a different core count.

Determinism: np.random is seeded from md5(source, oi, cadence[, model])
immediately before each fit. The SPL seed string is unchanged from
stage3_sf_linmix.py; BPL appends '_bpl' so the two models draw independently.
bpl_mcmc's emcee walker init (np.random.randn) is covered by the same seed.

Guards (both models): SIGALRM watchdog per fit; every exception caught per fit
and recorded as an invalid row; non-finite results demoted to invalid
(SF_linmix's 10**alpha can overflow to inf and would otherwise be 'valid').
"""

import os
import re
import sys
import glob
import json
import time
import signal
import hashlib
import contextlib
import numpy as np
import pandas as pd
from mpi4py import MPI

from sf_keys import keys_for
from newSF import SF_wnoise, SF_linmix, bpl_mcmc, parse_to_dict

# ── model selection ───────────────────────────────────────────────────────────
if "--bpl" in sys.argv:
    MODEL = "bpl"
elif "--spl" in sys.argv or "--linmix" in sys.argv:
    MODEL = "spl"
else:
    sys.exit("usage: mpirun -n N python stage3_sf_fit.py (--spl|--linmix|--bpl)")

# ── configuration ─────────────────────────────────────────────────────────────
N_SIMS      = 120
HOME        = os.environ["HOME"]

# NEW CAMPAIGN DIRECTORY. The 20-yr run redefines both 'full' (7306 even
# epochs vs the old per-source baseline) and 'crop' (ZTF pattern tiled over
# 20 yr vs mapped once). Reusing ~/results/sf_cache/ would silently mix two
# different experiments under identical labels, so this campaign is isolated.
OUTPUT_PATH = HOME + "/results/20yr_lc/sf_cache/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Cadence discovery table: name -> (directory, filename pattern).
# {tag} = "{ra}_{dec}_z{band}". A '*' in the pattern means several rank files
# are pooled. A cadence whose file(s) are absent for a source is SKIPPED for
# that source - no error, it simply yields fewer cadence rows.
# Edit the paths/patterns to match what stage 1 v3 / stage 2 v2 actually write.
# NOTE: 'crop' and 'ztfdur' share the filename sim_{tag}_cropped.parquet and
# are distinguished ONLY by directory. A wrong path here therefore fits the
# wrong light curve under the right label, silently - check these two lines
# before every campaign.
CADENCE_FILES = {
    "full":   (HOME + "/results/20yr_lc/full_lc/",    "sim_{tag}_rank*_full.parquet"),
    "crop":   (HOME + "/results/20yr_lc/cropped_lc/", "sim_{tag}_cropped.parquet"),
    "ztfdur": (HOME + "/results/20yr_lc/ZTF_dur/",    "sim_{tag}_cropped.parquet"),
}
CADENCES = tuple(CADENCE_FILES)        # fit order

# stage-1/2 sentinels gating which sources are ready
STAGE1_DIR = CADENCE_FILES["full"][0]
STAGE2_DIR = CADENCE_FILES["crop"][0]

MAX_BINS = 60    

FIT_KEYS = keys_for(MODEL)
KEY_FROM = {k: k for k in FIT_KEYS}

# BPL runs ~10-30x more likelihood evaluations than SPL -> longer watchdog
TIMEOUT = 300 if MODEL == "spl" else 900

BPL_P0 = [0.5, 0.5, 100]      # bpl_mcmc initial_guess [A, gamma, dt_break]

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
nrank = comm.Get_size()


# ── watchdog ──────────────────────────────────────────────────────────────────

class FitTimeout(Exception):
    pass

def _alarm(signum, frame):
    raise FitTimeout()


# ── paths ─────────────────────────────────────────────────────────────────────

def fit_jsonl(ra, dec, band):
    pre = "fit" if MODEL == "spl" else "fitbpl"
    return OUTPUT_PATH + f"{pre}_{ra}_{dec}_z{band}_rank{rank}.jsonl"

def sf_parquet(ra, dec, band):
    return OUTPUT_PATH + f"sf_{ra}_{dec}_z{band}_rank{rank}.parquet"

def sentinel_path(ra, dec, band):
    suf = "stage3" if MODEL == "spl" else "stage3bpl"
    return OUTPUT_PATH + f"sim_{ra}_{dec}_z{band}.{suf}.done"


def radec_band_from_sentinel(path, stage):
    m = re.search(rf"sim_(\d+\.\d+)_([-+]?\d+\.\d+)_z(\w)\.{stage}\.done",
                  os.path.basename(path))
    return (float(m.group(1)), float(m.group(2)), m.group(3)) if m else None


def fit_seed(ra, dec, band, oi, cadence):
    """SPL string unchanged from stage3_sf_linmix.py (preserves reproducibility
    of the existing cache); BPL gets its own independent stream."""
    s = f"{ra}_{dec}_{band}_{oi}_{cadence}"
    if MODEL == "bpl":
        s += "_bpl"
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def slice_bounds(rank):
    base, rem = divmod(N_SIMS, nrank)
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


# ── caches ────────────────────────────────────────────────────────────────────

def load_done_fits(path):
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((int(r["object_index"]), r["cadence"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_fit(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def load_sf_pool(ra, dec, band):
    """Pool ALL rank SF files for this source -> {(oi, cadence): row}.
    Pooling (not just this rank's file) makes SF reuse robust to running at a
    different core count than the run that produced the SF."""
    pool = {}
    for p in sorted(glob.glob(OUTPUT_PATH + f"sf_{ra}_{dec}_z{band}_rank*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        for _, r in df.iterrows():
            pool[(int(r["object_index"]), r["cadence"])] = r
    return pool


def sf_series_to_str(s):
    if len(s) >= MAX_BINS:
        raise ValueError(f"SF has {len(s)} bins >= {MAX_BINS}; to_string() would elide")
    return s.to_string()


def append_sf_rows(path, frames, rows):
    frames.extend(pd.DataFrame(r, index=[0]) for r in rows)
    out = pd.concat(frames, ignore_index=True)
    tmp = path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    frames.clear()
    frames.append(out)


def sf_dict_from_cache(srow):
    return {k: parse_to_dict(srow[k])["SF"] for k in ("SF", "SFmaxerr", "SFminerr")}


# ── the fit ───────────────────────────────────────────────────────────────────

def run_fit(sf_dict, ra, dec, band, oi, cadence):
    """Seed, arm the watchdog, run the selected model, validate.
    Returns (valid, fail_reason, values_dict)."""
    nan = {k: np.nan for k in FIT_KEYS}
    np.random.seed(fit_seed(ra, dec, band, oi, cadence))
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        if MODEL == "spl":
            fit = SF_linmix(sf_dict,path=OUTPUT_PATH+f'plots/{oi}_{cadence}_')            # mutates + returns sf_dict
        else:
            # bpl_mcmc indexes old_dict['RA'] directly and returns a NEW dict.
            # emcee's progress bar is hardcoded on -> silence stderr for the call.
            sf_dict["RA"] = ra
            with open(os.devnull, "w") as devnull, \
                    contextlib.redirect_stderr(devnull):
                # per-source PNG next to the output pkl (unique per input, avoids RA/band collisions)

                fit = bpl_mcmc(sf_dict, initial_guess=BPL_P0,path=OUTPUT_PATH+f'plots/{oi}_{cadence}_',
                               model_check=False, mcmc_check=False, plot=False,
                               progress=False, save_plot=False,verbose=False,model='at_break')
    except FitTimeout:
        print(f"[rank {rank}] TIMEOUT (> {TIMEOUT}s) {ra}_{dec}_z{band} "
              f"sim {oi} {cadence} [{MODEL}]", flush=True)
        return False, f"{MODEL} timeout (> {TIMEOUT}s)", nan
    except Exception as e:
        # bpl_mcmc's first curve_fit is unwrapped -> RuntimeError lands here
        return False, f"{type(e).__name__}: {e}", nan
    finally:
        signal.alarm(0)

    if fit is None:
        return False, f"{MODEL} returned None", nan
    try:
        vals = {k: float(fit[KEY_FROM[k]]) for k in FIT_KEYS}
    except KeyError as e:
        return False, f"missing key {e}", nan
    if not all(np.isfinite(v) for v in vals.values()):
        # SF_linmix's 10**alpha can overflow to inf; such a fit is not usable
        return False, "non-finite fit values", vals
    return True, "", vals


# ── light curves (only needed when SF is not cached) ──────────────────────────

def load_cadence_lc(ra, dec, band, cadence):
    """Load the LC for one cadence, pooling rank files if the pattern globs.
    Returns None when the cadence does not exist for this source."""
    directory, pattern = CADENCE_FILES[cadence]
    tag = f"{ra}_{dec}_z{band}"
    files = sorted(glob.glob(directory + pattern.format(tag=tag)))
    if not files:
        return None
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def available_cadences(ra, dec, band):
    """Which cadences have LC files on disk for this source."""
    out = []
    for cad in CADENCES:
        directory, pattern = CADENCE_FILES[cad]
        if glob.glob(directory + pattern.format(tag=f"{ra}_{dec}_z{band}")):
            out.append(cad)
    return out


# ── per-source ────────────────────────────────────────────────────────────────

def process_source(ra, dec, band):
    start, stop = slice_bounds(rank)
    if stop <= start:
        return True

    fpath = fit_jsonl(ra, dec, band)
    spath = sf_parquet(ra, dec, band)
    done  = load_done_fits(fpath)
    pool  = load_sf_pool(ra, dec, band)

    # cadences present on disk (plus any already represented in the SF cache,
    # so a cadence whose LC was cleaned up can still be re-fitted from cache)
    cads = available_cadences(ra, dec, band)
    cached_cads = {c for (_, c) in pool}
    cads = [c for c in CADENCES if c in cads or c in cached_cads]
    if not cads:
        print(f"[rank {rank}] {ra}_{dec}_z{band}: no cadence LCs or cached SF",
              flush=True)
        return False
    missing = [c for c in CADENCES if c not in cads]
    if missing and rank == 0:
        print(f"[rank 0] {ra}_{dec}_z{band}: cadences present {cads}, "
              f"absent {missing} (skipped)", flush=True)

    lcs = {}          # cadence -> DataFrame, loaded lazily only if SF missing

    sf_frames = []
    if os.path.exists(spath):
        sf_frames = [pd.read_parquet(spath)]
    have_own = set()
    if sf_frames:
        have_own = set(zip(sf_frames[0]["object_index"], sf_frames[0]["cadence"]))

    t0, n, reused = time.time(), 0, 0
    for oi in range(start, stop):
        for cadence in cads:
            if (oi, cadence) in done:
                continue

            if (oi, cadence) in pool:
                sf_dict = sf_dict_from_cache(pool[(oi, cadence)])
                reused += 1
            else:
                if cadence not in lcs:
                    lcs[cadence] = load_cadence_lc(ra, dec, band, cadence)
                lc = lcs[cadence]
                if lc is None:
                    print(f"[rank {rank}] {ra}_{dec}_z{band} sim {oi} "
                          f"{cadence}: SF not cached and LC absent", flush=True)
                    continue
                g = lc[lc.object_index == oi].sort_values("time")
                if len(g) == 0:
                    print(f"[rank {rank}] {ra}_{dec}_z{band} sim {oi} "
                          f"{cadence}: absent in LC input", flush=True)
                    continue
                sf_dict = SF_wnoise(g["mag"].values, g["time"].values,
                                    g["err"].values, cs_all=None,
                                    color="#436BAD", verbose=False)[0]
                if (oi, cadence) not in have_own:
                    append_sf_rows(spath, sf_frames, [{
                        "object_index": oi, "cadence": cadence,
                        "SF":       sf_series_to_str(sf_dict["SF"]),
                        "SFmaxerr": sf_series_to_str(sf_dict["SFmaxerr"]),
                        "SFminerr": sf_series_to_str(sf_dict["SFminerr"]),
                    }])
                    have_own.add((oi, cadence))

            valid, reason, vals = run_fit(sf_dict, ra, dec, band, oi, cadence)
            append_fit(fpath, {"object_index": oi, "cadence": cadence,
                               "model": MODEL, "valid": valid,
                               "fail_reason": reason, **vals})
            done.add((oi, cadence))
            n += 1

        el = time.time() - t0
        print(f"[rank {rank}] {ra}_{dec}_z{band} sim {oi} [{MODEL}] "
              f"({n} fits, {el:.1f}s, {el/max(n,1):.1f}s/fit, "
              f"{reused} SF reused)", flush=True)

    # free the big 20-yr frames before the next source
    lcs.clear()

    print(f"[rank {rank}] {ra}_{dec}_z{band} slice [{start}:{stop}) [{MODEL}] "
          f"{n} new fits over {len(cads)} cadences {time.time()-t0:.1f}s "
          f"({reused} SF reused)", flush=True)
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    job_start = time.time()

    if rank == 0:
        s1 = {radec_band_from_sentinel(p, "stage1")
              for p in glob.glob(STAGE1_DIR + "*.stage1.done")}
        s2 = {radec_band_from_sentinel(p, "stage2")
              for p in glob.glob(STAGE2_DIR + "*.stage2.done")}
        ready = sorted(x for x in (s1 & s2) if x is not None)
        todo = [s for s in ready if not os.path.exists(sentinel_path(*s))]
        print(f"[rank 0] model={MODEL} timeout={TIMEOUT}s cadences={list(CADENCES)} "
              f"cache={OUTPUT_PATH}", flush=True)
        print(f"[rank 0] {len(ready)} ready, {len(todo)} to process", flush=True)
    else:
        todo = None
    todo = comm.bcast(todo, root=0)

    for ra, dec, band in todo:
        comm.Barrier()
        t0 = time.time()
        try:
            ok = process_source(ra, dec, band)
        except Exception as e:
            print(f"[rank {rank}] ERROR {ra}_{dec}_z{band}: "
                  f"{type(e).__name__}: {e}", flush=True)
            ok = False
        all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN)
        comm.Barrier()
        if rank == 0:
            if all_ok:
                open(sentinel_path(ra, dec, band), "w").close()
                print(f"[rank 0] {ra}_{dec}_z{band} [{MODEL}] SOURCE DONE "
                      f"{time.time()-t0:.1f}s", flush=True)
            else:
                print(f"[rank 0] {ra}_{dec}_z{band} [{MODEL}] FAILED "
                      f"(no sentinel)", flush=True)

    comm.Barrier()
    if rank == 0:
        print(f"[rank 0] stage 3 [{MODEL}] walltime "
              f"{time.time()-job_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()