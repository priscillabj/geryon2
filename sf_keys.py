"""
sf_keys.py — single source of truth for stage-3 cache key sets.

Imported by stage3_sf_fit.py and every checker/merge/plot script so the key
vocabulary is defined exactly once. Deliberately has NO imports (no mpi4py, no
argv parsing) so it is safe to import from serial tools.

The two models are NOT a suffix swap of one another. Every amplitude now
carries its OWN explicitly-named error, so there is no ambiguous 'A_maxerr':
  * SPL reports A_1 (dt=1 intercept) and A_365 (amplitude at 365 d), each with
    its own A_1_(min/max)err_spl and A_365_(min/max)err_spl.
  * BPL reports A_1 (dt=1 intercept) and A_break (plateau = A_1*dt_break**gamma)
    and dt_break, each with its own explicitly-named errors.
so each list is written out explicitly rather than templated.
"""

# ── current campaign (unprefixed names, newSF.py as of the 20-yr run) ─────────

SPL_KEYS = [
    "A_1_spl", "A_1_maxerr_spl", "A_1_minerr_spl",
    "A_365_spl", "A_365_maxerr_spl", "A_365_minerr_spl",
    "gamma_spl", "gamma_maxerr_spl", "gamma_minerr_spl",
]

BPL_KEYS = [
    "A_1_bpl", "A_1_maxerr_bpl", "A_1_minerr_bpl",
    "A_break_bpl", "A_break_maxerr_bpl", "A_break_minerr_bpl",
    "gamma_bpl", "gamma_maxerr_bpl", "gamma_minerr_bpl",
    "dt_break_bpl", "dt_break_maxerr_bpl", "dt_break_minerr_bpl",
]

# NOTE: errors are now name-scoped to their amplitude (A_1_maxerr_spl vs
# A_365_maxerr_spl; A_1_maxerr_bpl vs A_break_maxerr_bpl), so there is no
# generic 'A_maxerr' to confuse across models. The SPL A_365 amplitude and the
# BPL A_break amplitude are still different quantities - don't pair them.

# ── starplaw (SF_boot.py): SF_obs = sqrt(PL^2 + (N SF_star)^2), joint bootstrap ──
# Fitted to the observed SF with the comparison-star template as a noise term
# (no noise subtraction). A_L is the power-law amplitude at L days, pivot-free:
# A_365_boot is the one to put next to A_365_spl. N scales the star template.
# Values are medians, errors 16/84-percentile distances, of the bootstrap MAP
# refits. Then fit diagnostics/flags and run settings; 'epochs', 'n_stars' and
# 'nboot' are unsuffixed in the pkl (merge's extract() suffixes them).

BOOT_KEYS = [
    "A_1_boot", "A_1_maxerr_boot", "A_1_minerr_boot",
    "A_100_boot", "A_100_maxerr_boot", "A_100_minerr_boot",
    "A_300_boot", "A_300_maxerr_boot", "A_300_minerr_boot",
    "A_365_boot", "A_365_maxerr_boot", "A_365_minerr_boot",
    "gamma_boot", "gamma_maxerr_boot", "gamma_minerr_boot",
    "N_boot", "N_maxerr_boot", "N_minerr_boot",
    "SF_pivot_boot", "SF_pivot_maxerr_boot", "SF_pivot_minerr_boot", "pivot_boot",
    "n_bins_boot", "dof_boot", "chi2_full_boot", "chi2red_boot",
    "frac_above_model_boot", "adjcorr_boot", "frac_intr_maxlag_boot",
    "flag_boot", "N_prior_dominated_boot",
    "method_boot", "nboot", "epochs", "n_stars",
]

# ── legacy campaign (leading '_' from the empty `phot` prefix) ────────────────
# ~/results/sf_cache/ (the 1018-source SPL run) was written with these names.
# Point tools at LEGACY_SPL_KEYS when reading that cache.

LEGACY_SPL_KEYS = [
    "_A_1", "_A_365_spl", "_A_maxerr_spl", "_A_minerr_spl",
    "_gamma_spl", "_gamma_maxerr_spl", "_gamma_minerr_spl",
]

KEYS = {
    "spl":        SPL_KEYS,
    "bpl":        BPL_KEYS,
    "boot":       BOOT_KEYS,
    "spl_legacy": LEGACY_SPL_KEYS,
}

# the slope column per model — what stage 5 averages and the checkers report
GAMMA = {
    "spl":        "gamma_spl",
    "bpl":        "gamma_bpl",
    "boot":       "gamma_boot",
    "spl_legacy": "_gamma_spl",
}

# non-key columns every stage-3 JSONL row carries
META_KEYS = ["object_index", "cadence", "model", "valid", "fail_reason"]


def keys_for(model):
    """Canonical FIT_KEYS list for a model name. Raises on typos rather than
    returning a plausible-but-wrong list."""
    try:
        return list(KEYS[model])
    except KeyError:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(KEYS)}")


def gamma_for(model):
    try:
        return GAMMA[model]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(GAMMA)}")