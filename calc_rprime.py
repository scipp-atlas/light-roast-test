#!/usr/bin/env python3
"""
Calculate R' = N_TT * N_LL / (N_LT * N_TL) for JFP+Other background MC,
and the data/MC scale factor SF = (data_TT - prompt_TT) / DD_estimate,
where DD_estimate = (data_TL-prompt_TL)*(data_LT-prompt_LT)/(data_LL-prompt_LL).

For R': per-sample values are computed from background MC, then combined via
an uncertainty-weighted average.

For SF: data counts are summed over all data files for the era; MC prompt
contributions (real+EFP+unclassified) are summed over all background MC files;
SF = prompt-subtracted data in TT / data-driven DD estimate from sidebands.

Results are produced separately for Run 2 and Run 3, for LoosePrime4 and Loose.
Reads pre-computed JSON files from ABCD_results_4/.
"""

import argparse
import glob
import json
import math
import os
import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "ABCD_results_4")
LOOSE_PRIMES = ["LoosePrime4", "Loose"]
REGIONS = [
    ("Preselection", "0L"),
    ("VR",           "0L-mT-mid"),
    ("SR",           "0L-mT-low"),
    ("SR",           "0L-mT-mid"),
    ("SR",           "0L-mT-hgh"),
]

SKIP_TAGS    = ("gammajet", "jetjet", "_jj_", "N2_", "signal", "data_")
RUN2_YEARS   = {2015, 2016, 2017, 2018}
RUN3_YEARS   = {2022, 2023, 2024}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_run2_mc(path):
    return "mc20" in os.path.basename(path)

def is_run3_mc(path):
    return "mc23" in os.path.basename(path)

def jfp_other(b, jfp_only=False):
    """(sumweights, staterr) for JFP+Other (or JFP only) in one ABCD bin dict."""
    if jfp_only:
        return b["jfp"]["sumweights"], b["jfp"]["staterr"]
    sw = b["jfp"]["sumweights"] + b["other"]["sumweights"]
    se = math.sqrt(b["jfp"]["staterr"]**2 + b["other"]["staterr"]**2)
    return sw, se

def prompt(b):
    """(sumweights, staterr) for prompt (real+EFP+unclassified) in one bin."""
    sw = b["real"]["sumweights"] + b["efp"]["sumweights"] + b["unclassified"]["sumweights"]
    se = math.sqrt(b["real"]["staterr"]**2 + b["efp"]["staterr"]**2 + b["unclassified"]["staterr"]**2)
    return sw, se

def calc_rprime(n_tt, n_tl, n_lt, n_ll):
    denom = n_lt * n_tl
    if denom == 0 or n_tt == 0 or n_ll == 0:
        return float("nan")
    return n_tt * n_ll / denom

def calc_rprime_err(rp, n_tt, e_tt, n_tl, e_tl, n_lt, e_lt, n_ll, e_ll):
    if rp == 0 or math.isnan(rp):
        return float("nan")
    rel2 = (e_tt/n_tt)**2 + (e_tl/n_tl)**2 + (e_lt/n_lt)**2 + (e_ll/n_ll)**2
    return rp * math.sqrt(rel2)

def weighted_average(samples):
    """Uncertainty-weighted average of (value, sigma) pairs."""
    if not samples:
        return float("nan"), float("nan"), 0
    weights   = [1.0 / s**2 for _, s in samples]
    sum_w     = sum(weights)
    avg       = sum(w * v for w, (v, _) in zip(weights, samples)) / sum_w
    sigma_avg = 1.0 / math.sqrt(sum_w)
    return avg, sigma_avg, len(samples)

# ---------------------------------------------------------------------------
# R' from per-sample MC
# ---------------------------------------------------------------------------

def collect_rprime_totals(loose_prime, run2, jfp_only=False):
    """
    Sum JFP+Other (or JFP only) yields across all background MC samples, then compute R'
    and its propagated uncertainty from those totals.
    Returns dict region_key -> (rp, sigma).
    """
    tot_n  = {f"{rt}/{rn}": {b: 0.0 for b in ("TT","TL","LT","LL")} for rt, rn in REGIONS}
    tot_e2 = {f"{rt}/{rn}": {b: 0.0 for b in ("TT","TL","LT","LL")} for rt, rn in REGIONS}

    pattern = os.path.join(RESULTS_DIR,
                           f"output_*_ABCD_tightID_hybridCOIso_{loose_prime}.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if any(t in fname for t in SKIP_TAGS):
            continue
        if run2 and not is_run2_mc(path): continue
        if not run2 and not is_run3_mc(path): continue
        with open(path) as f:
            data = json.load(f)
        for reg_type, reg_name in REGIONS:
            key = f"{reg_type}/{reg_name}"
            try:
                reg = data[reg_type][reg_name]
            except KeyError:
                continue
            for b in ("TT", "TL", "LT", "LL"):
                sw, se = jfp_other(reg[b], jfp_only=jfp_only)
                tot_n[key][b]  += sw
                tot_e2[key][b] += se**2

    results = {}
    for key in tot_n:
        n  = tot_n[key]
        e2 = tot_e2[key]
        rp    = calc_rprime(n["TT"], n["TL"], n["LT"], n["LL"])
        sigma = calc_rprime_err(rp,
                                n["TT"], math.sqrt(e2["TT"]),
                                n["TL"], math.sqrt(e2["TL"]),
                                n["LT"], math.sqrt(e2["LT"]),
                                n["LL"], math.sqrt(e2["LL"]))
        results[key] = (rp, sigma)
    return results

# ---------------------------------------------------------------------------
# SF from data vs. data-driven estimate
# ---------------------------------------------------------------------------

def _zero_abcd():
    """Zero-initialised per-bin accumulators for data and MC prompt."""
    return {b: {"data": 0, "prompt_sw": 0.0, "prompt_se2": 0.0}
            for b in ("TT", "TL", "LT", "LL")}

def collect_sf(loose_prime, run2):
    """
    Accumulate:
      - data counts from all data files for the era
      - MC prompt (real+EFP+unclassified) sumweights from all background MC files

    Returns dict region_key -> (sf, sf_err) or (nan, nan) if insufficient counts.

    SF = (data_TT - prompt_TT) / DD_estimate
    where DD_estimate = (data_TL-prompt_TL)*(data_LT-prompt_LT)/(data_LL-prompt_LL)
    """
    acc = {f"{rt}/{rn}": _zero_abcd() for rt, rn in REGIONS}

    pattern = os.path.join(RESULTS_DIR,
                           f"output_*_ABCD_tightID_hybridCOIso_{loose_prime}.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        basename = os.path.basename(path)

        # ---- data files ----
        if "data_" in fname:
            try:
                year = int(basename.split("_")[2])
            except (IndexError, ValueError):
                continue
            if run2  and year not in RUN2_YEARS: continue
            if not run2 and year not in RUN3_YEARS: continue
            with open(path) as f:
                d = json.load(f)
            for reg_type, reg_name in REGIONS:
                key = f"{reg_type}/{reg_name}"
                try:
                    reg = d[reg_type][reg_name]
                except KeyError:
                    continue
                for b in ("TT", "TL", "LT", "LL"):
                    acc[key][b]["data"] += reg[b]["data"]
            continue

        # ---- background MC files ----
        if any(t in fname for t in SKIP_TAGS):
            continue
        if run2 and not is_run2_mc(path): continue
        if not run2 and not is_run3_mc(path): continue
        with open(path) as f:
            d = json.load(f)
        for reg_type, reg_name in REGIONS:
            key = f"{reg_type}/{reg_name}"
            try:
                reg = d[reg_type][reg_name]
            except KeyError:
                continue
            for b in ("TT", "TL", "LT", "LL"):
                sw, se = prompt(reg[b])
                acc[key][b]["prompt_sw"]  += sw
                acc[key][b]["prompt_se2"] += se**2

    # Compute SF per region
    results = {}
    for key, bins in acc.items():
        # prompt-subtracted counts and their uncertainties
        n, e2 = {}, {}
        for b in ("TT", "TL", "LT", "LL"):
            n[b]  = bins[b]["data"] - bins[b]["prompt_sw"]
            # uncertainty: Poisson on data + MC prompt stat
            e2[b] = bins[b]["data"] + bins[b]["prompt_se2"]

        denom = n["TL"] * n["LT"]
        if n["LL"] <= 0 or denom <= 0 or n["TT"] <= 0:
            results[key] = (float("nan"), float("nan"))
            continue

        dd_est = n["TL"] * n["LT"] / n["LL"]

        # uncertainty on DD estimate: σ_DD/DD = sqrt(σ_TL²/N_TL² + σ_LT²/N_LT² + σ_LL²/N_LL²)
        dd_err = dd_est * math.sqrt(e2["TL"]/n["TL"]**2 +
                                    e2["LT"]/n["LT"]**2 +
                                    e2["LL"]/n["LL"]**2)

        sf     = n["TT"] / dd_est
        sf_err = sf * math.sqrt(e2["TT"]/n["TT"]**2 + (dd_err/dd_est)**2)

        results[key] = (sf, sf_err)

    return results

# ---------------------------------------------------------------------------
# VR data vs. prediction comparison
# ---------------------------------------------------------------------------

def collect_vr_comparison(loose_prime, run2, jfp_only=False):
    """
    For VR-0L-mT-mid accumulate:
      - data counts in all ABCD bins
      - MC subtraction in each bin:
          default  : real + EFP + unclassified  (keeps JFP+Other in sidebands)
          jfp_only : real + EFP + unclassified + Other  (keeps JFP only)
      - Total MC yield in TT (all truth categories summed)

    Returns a dict with keys:
      data_n      : {TT,TL,LT,LL} raw integer counts
      sub_sw      : {TT,TL,LT,LL} sumweights to subtract
      sub_se2     : {TT,TL,LT,LL} variance on the subtraction
      mc_tt_sw    : total MC sumweights in TT (all categories)
      mc_tt_se2   : variance on mc_tt_sw
      mc_tt_cat   : {"prompt","efp","jfp","other"} sumweights in TT
      mc_tt_cat_e2: {"prompt","efp","jfp","other"} variance in TT
    """
    REG_TYPE, REG_NAME = "VR", "0L-mT-mid"
    data_n       = {b: 0   for b in ("TT", "TL", "LT", "LL")}
    sub_sw       = {b: 0.0 for b in ("TT", "TL", "LT", "LL")}
    sub_se2      = {b: 0.0 for b in ("TT", "TL", "LT", "LL")}
    mc_tt_sw     = 0.0
    mc_tt_se2    = 0.0
    mc_tt_cat    = {c: 0.0 for c in ("prompt", "efp", "jfp", "other")}
    mc_tt_cat_e2 = {c: 0.0 for c in ("prompt", "efp", "jfp", "other")}

    pattern = os.path.join(RESULTS_DIR,
                           f"output_*_ABCD_tightID_hybridCOIso_{loose_prime}.json")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)

        # ---- data files ----
        if "data_" in fname:
            try:
                year = int(fname.split("_")[2])
            except (IndexError, ValueError):
                continue
            era_years = RUN2_YEARS if run2 else RUN3_YEARS
            if year not in era_years:
                continue
            with open(path) as f:
                d = json.load(f)
            try:
                reg = d[REG_TYPE][REG_NAME]
            except KeyError:
                continue
            for b in ("TT", "TL", "LT", "LL"):
                data_n[b] += reg[b]["data"]
            continue

        # ---- background MC files ----
        if any(t in fname for t in SKIP_TAGS):
            continue
        if run2 and not is_run2_mc(path): continue
        if not run2 and not is_run3_mc(path): continue
        with open(path) as f:
            d = json.load(f)
        try:
            reg = d[REG_TYPE][REG_NAME]
        except KeyError:
            continue

        for b in ("TT", "TL", "LT", "LL"):
            bin_d = reg[b]
            sw, se = prompt(bin_d)
            if jfp_only:
                sw += bin_d["other"]["sumweights"]
                se  = math.sqrt(se**2 + bin_d["other"]["staterr"]**2)
            sub_sw[b]  += sw
            sub_se2[b] += se**2

        # Total MC TT: sum all truth categories + per-category breakdown
        bin_tt = reg["TT"]
        for cat in ("real", "efp", "jfp", "other", "unclassified"):
            mc_tt_sw  += bin_tt[cat]["sumweights"]
            mc_tt_se2 += bin_tt[cat]["staterr"]**2
        # prompt = real + unclassified (EFP tracked separately)
        for raw_cat, grp in (("real", "prompt"), ("unclassified", "prompt"),
                              ("efp", "efp"), ("jfp", "jfp"), ("other", "other")):
            mc_tt_cat[grp]    += bin_tt[raw_cat]["sumweights"]
            mc_tt_cat_e2[grp] += bin_tt[raw_cat]["staterr"]**2

    return dict(data_n=data_n, sub_sw=sub_sw, sub_se2=sub_se2,
                mc_tt_sw=mc_tt_sw, mc_tt_se2=mc_tt_se2,
                mc_tt_cat=mc_tt_cat, mc_tt_cat_e2=mc_tt_cat_e2)


def print_vr_data_comparison(loose_prime, rprime_results, run2, jfp_only=False):
    """
    Print three quantities for VR-0L-mT-mid TT, side by side:
      a) Data yield
      b) Full MC prediction (all truth categories)
      c) Hybrid: MC prompt + MC EFP  +  ABCD estimate for JFP+Other (or JFP only)
         Scale factor = data_TT / hybrid_total

    Sideband subtraction for ABCD: data − (prompt + EFP) to isolate JFP+Other fakes.
    With --jfp-only: subtract also Other, so ABCD estimates JFP alone.
    """
    era = "Run 2" if run2 else "Run 3"
    rk  = "VR/0L-mT-mid"
    rp, rp_err = rprime_results.get(rk, (float("nan"), float("nan")))

    comp         = collect_vr_comparison(loose_prime, run2, jfp_only)
    data_n       = comp["data_n"]
    sub_sw       = comp["sub_sw"]   # prompt+EFP (or +Other) MC subtraction
    sub_se2      = comp["sub_se2"]
    mc_tt        = comp["mc_tt_sw"]
    mc_tt_e      = math.sqrt(comp["mc_tt_se2"])
    mc_tt_cat    = comp["mc_tt_cat"]
    mc_tt_cat_e2 = comp["mc_tt_cat_e2"]

    # Sideband fake yields: data minus prompt+EFP MC (captures JFP+Other fakes)
    fake    = {}
    fake_e2 = {}
    for b in ("TL", "LT", "LL"):
        fake[b]    = data_n[b] - sub_sw[b]
        fake_e2[b] = data_n[b] + sub_se2[b]   # Poisson on data + MC stat

    abcd_label = "JFP" if jfp_only else "JFP+Other"
    sub_label  = "prompt+EFP+Other" if jfp_only else "prompt+EFP"

    print(f"\n  {'═'*70}")
    print(f"  {loose_prime}  |  VR 0L-mT-mid  |  {era}")
    print(f"  {'═'*70}")

    # Show sideband inputs
    print(f"\n  Sideband fake yields (data − MC {sub_label}):")
    for b in ("TL", "LT", "LL"):
        fake_e = math.sqrt(fake_e2[b])
        print(f"    fake {b}  {fake[b]:>10.2f}  ±{fake_e:>8.2f}")

    ok = fake["LL"] > 0 and fake["TL"] > 0 and fake["LT"] > 0
    if not ok:
        print("    *** insufficient sideband yields — cannot form ABCD estimate ***")
        return

    # ABCD estimate for JFP+Other (or JFP) fakes in TT
    dd_est = fake["TL"] * fake["LT"] / fake["LL"]
    dd_err = dd_est * math.sqrt(
        fake_e2["TL"] / fake["TL"]**2 +
        fake_e2["LT"] / fake["LT"]**2 +
        fake_e2["LL"] / fake["LL"]**2
    )
    if not math.isnan(rp) and rp > 0:
        abcd_est     = rp * dd_est
        abcd_est_err = abcd_est * math.sqrt((rp_err / rp)**2 + (dd_err / dd_est)**2)
    else:
        abcd_est, abcd_est_err = float("nan"), float("nan")

    # MC prompt and EFP in TT (used in hybrid estimate)
    mc_prompt     = mc_tt_cat["prompt"]
    mc_prompt_e   = math.sqrt(mc_tt_cat_e2["prompt"])
    mc_efp        = mc_tt_cat["efp"]
    mc_efp_e      = math.sqrt(mc_tt_cat_e2["efp"])

    # Hybrid total = MC prompt + MC EFP + ABCD(JFP+Other)
    if not math.isnan(abcd_est):
        hybrid     = mc_prompt + mc_efp + abcd_est
        hybrid_e2  = mc_tt_cat_e2["prompt"] + mc_tt_cat_e2["efp"] + abcd_est_err**2
        hybrid_err = math.sqrt(hybrid_e2)
    else:
        hybrid, hybrid_err = float("nan"), float("nan")

    data_tt   = data_n["TT"]
    data_tt_e = math.sqrt(data_tt) if data_tt > 0 else 0.0

    # Scale factor: data / hybrid
    if not math.isnan(hybrid) and hybrid > 0:
        sf     = data_tt / hybrid
        sf_err = sf * math.sqrt(data_tt_e**2 / data_tt**2 + (hybrid_err / hybrid)**2)
    else:
        sf, sf_err = float("nan"), float("nan")

    w = 10
    print()
    hdr = f"  {'Quantity':<44}  {'Value':>{w}}  {'Unc':>{w}}"
    print(hdr)
    print(f"  {'─'*70}")

    def row(label, val, err=None):
        val_s = f"{val:>{w}.2f}"
        err_s = f"±{err:>{w-1}.2f}" if err is not None else f"{'':>{w+1}}"
        print(f"  {label:<44}  {val_s}  {err_s}")

    # (a) Data
    row("(a) Data TT",                               data_tt,      data_tt_e)
    print(f"  {'─'*70}")

    # (b) Full MC
    row("(b) Full MC TT",                            mc_tt,        mc_tt_e)
    row("      prompt (real+unclassified)",           mc_prompt,    mc_prompt_e)
    row("      EFP",                                  mc_efp,       mc_efp_e)
    row("      JFP",                                  mc_tt_cat["jfp"],
                                                      math.sqrt(mc_tt_cat_e2["jfp"]))
    row("      Other",                                mc_tt_cat["other"],
                                                      math.sqrt(mc_tt_cat_e2["other"]))
    print(f"  {'─'*70}")

    # (c) Hybrid
    rp_tag = f"R'={rp:.4f}±{rp_err:.4f}" if not math.isnan(rp) else "R' unavail."
    row(f"(c) Hybrid MC TT  [{rp_tag}]",             hybrid,       hybrid_err)
    row("      MC prompt",                            mc_prompt,    mc_prompt_e)
    row("      MC EFP",                               mc_efp,       mc_efp_e)
    row(f"      ABCD est ({abcd_label}, R'-scaled)",  abcd_est,     abcd_est_err)
    print(f"  {'─'*70}")
    row("SF  (a) / (c)  [data / hybrid]",             sf,           sf_err)
    print(f"  {'─'*70}")


# ---------------------------------------------------------------------------
# Sample grouping
# ---------------------------------------------------------------------------

_FILTER_RE = re.compile(r'_(CVetoBVeto|BFilter|CFilterBVeto)(_mc\d+\w*)$')

def group_key(sample):
    """
    Strip known HF-filter suffixes (CVetoBVeto, BFilter, CFilterBVeto) so that
    sliced samples of the same process are merged into one row.
    e.g. Sh_2214_Znunu_pTV2_CVetoBVeto_mc23  →  Sh_2214_Znunu_pTV2_mc23
         Sh_2214_Znunu_pTV2_BFilter_mc23      →  Sh_2214_Znunu_pTV2_mc23
    Samples without a recognised filter suffix are left unchanged.
    """
    m = _FILTER_RE.search(sample)
    if m:
        return sample[:m.start()] + m.group(2)
    return sample


# ---------------------------------------------------------------------------
# Diagnostic: per-sample yields table
# ---------------------------------------------------------------------------

def print_sample_table(loose_prime, reg_type, reg_name, run2, jfp_only=False):
    """
    Print a per-sample table of JFP+Other (or JFP only) yields in each ABCD bin
    and the resulting per-sample R', for a given region and LoosePrime working point.
    Also prints the totals and the weighted-average R'.
    """
    era  = "Run 2" if run2 else "Run 3"
    key  = f"{reg_type}/{reg_name}"
    pattern = os.path.join(RESULTS_DIR,
                           f"output_*_ABCD_tightID_hybridCOIso_{loose_prime}.json")

    # Accumulate per-sample yields, keyed by group label
    from collections import defaultdict
    grp_n  = defaultdict(lambda: {b: 0.0 for b in ("TT","TL","LT","LL")})
    grp_e2 = defaultdict(lambda: {b: 0.0 for b in ("TT","TL","LT","LL")})

    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if any(t in fname for t in SKIP_TAGS):
            continue
        if run2 and not is_run2_mc(path): continue
        if not run2 and not is_run3_mc(path): continue
        with open(path) as f:
            d = json.load(f)
        try:
            reg = d[reg_type][reg_name]
        except KeyError:
            continue
        sample = fname.replace("output_", "").replace(
            f"_ABCD_tightID_hybridCOIso_{loose_prime}.json", "")
        gkey = group_key(sample)
        for b in ("TT", "TL", "LT", "LL"):
            sw, se = jfp_other(reg[b], jfp_only=jfp_only)
            grp_n[gkey][b]  += sw
            grp_e2[gkey][b] += se**2

    # Build rows from grouped totals
    rows = []
    for gkey, n in grp_n.items():
        if n["TT"] + n["TL"] + n["LT"] + n["LL"] == 0:
            continue
        e2    = grp_e2[gkey]
        rp    = calc_rprime(n["TT"], n["TL"], n["LT"], n["LL"])
        sigma = calc_rprime_err(rp,
                                n["TT"], math.sqrt(e2["TT"]),
                                n["TL"], math.sqrt(e2["TL"]),
                                n["LT"], math.sqrt(e2["LT"]),
                                n["LL"], math.sqrt(e2["LL"])) \
                if not math.isnan(rp) else float("nan")
        rows.append((gkey, n["TT"], n["TL"], n["LT"], n["LL"], rp, sigma))

    print(f"\n{'='*110}")
    print(f"  {loose_prime}  |  {key}  |  {era}")
    print(f"{'='*110}")
    rp_col, srp_col = "R'", "σ(R')"
    hdr = f"  {'Sample':<55}  {'N_TT':>7}  {'N_TL':>7}  {'N_LT':>7}  {'N_LL':>7}  {rp_col:>8}  {srp_col:>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows.sort(key=lambda r: r[1], reverse=True)  # sort by N_TT descending

    tot = {b: 0.0 for b in ("TT", "TL", "LT", "LL")}
    rp_samples = []
    for sample, n_tt, n_tl, n_lt, n_ll, rp, sigma in rows:
        rp_str    = f"{rp:.4f}"    if not math.isnan(rp)    else "    ---"
        sigma_str = f"{sigma:.4f}" if not math.isnan(sigma) else "    ---"
        print(f"  {sample:<55}  {n_tt:>7.2f}  {n_tl:>7.2f}  {n_lt:>7.2f}  {n_ll:>7.2f}  {rp_str:>8}  {sigma_str:>8}")
        tot["TT"] += n_tt; tot["TL"] += n_tl; tot["LT"] += n_lt; tot["LL"] += n_ll
        if not math.isnan(rp) and not math.isnan(sigma) and sigma > 0:
            rp_samples.append((rp, sigma))

    print("  " + "-" * (len(hdr) - 2))
    rp_tot   = calc_rprime(tot["TT"], tot["TL"], tot["LT"], tot["LL"])
    print(f"  {'TOTAL':<55}  {tot['TT']:>7.2f}  {tot['TL']:>7.2f}  {tot['LT']:>7.2f}  {tot['LL']:>7.2f}  {rp_tot:>8.4f}")
    rp_avg, rp_avg_err, n_samp = weighted_average(rp_samples)
    print(f"\n  Weighted-average R' ({n_samp} samples): {rp_avg:.4f} ± {rp_avg_err:.4f}")
    print(f"  R' from summed totals:               {rp_tot:.4f}")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

REGION_LABELS = {
    "Preselection/0L":  "Preselection 0L",
    "VR/0L-mT-mid":     "VR 0L-mT-mid",
    "SR/0L-mT-low":     "SR 0L-mT-low",
    "SR/0L-mT-mid":     "SR 0L-mT-mid",
    "SR/0L-mT-hgh":     "SR 0L-mT-high",
}

REGION_LABELS_TEX = {
    "Preselection/0L":  r"Preselection 0L",
    "VR/0L-mT-mid":     r"VR 0L-$m_T$ mid",
    "SR/0L-mT-low":     r"SR 0L-$m_T$ low",
    "SR/0L-mT-mid":     r"SR 0L-$m_T$ mid",
    "SR/0L-mT-hgh":     r"SR 0L-$m_T$ high",
}

LP_LABELS_TEX = {
    "LoosePrime4": r"\texttt{LoosePrime4}",
    "Loose":       r"\texttt{Loose}",
}


def fmt_val(v, e):
    if math.isnan(v): return "        ---        "
    return f"{v:.4f} ± {e:.4f}"

def fmt_cell_tex(v, e, n=None):
    if math.isnan(v):
        return r"\multicolumn{1}{c}{---}"
    s = rf"${v:.3f} \pm {e:.3f}$"
    if n is not None:
        s += f" ({n})"
    return s


def make_latex_table(rprime_results, sf_results, jfp_only=False):
    """
    rprime_results[lp][era][rk] = (rp_avg, sigma_avg, n_samples)
    sf_results[lp][era][rk]     = (sf, sf_err)
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    mc_label = "JFP" if jfp_only else "JFP+Other"
    lines.append(r"\caption{%")
    lines.append(rf"  $R' = N_{{TT}} N_{{LL}} / (N_{{LT}} N_{{TL}})$ computed from {mc_label} yields")
    lines.append(r"  summed across all background MC samples, and data/MC scale factor SF\@.}")
    lines.append(r"\label{tab:rprime_sf}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Loose ID & Region"
                 r" & \multicolumn{2}{c}{$R'$ (MC)} "
                 r" & \multicolumn{2}{c}{SF (data)} \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    lines.append(r"& & Run 2 & Run 3 & Run 2 & Run 3 \\")
    lines.append(r"\midrule")

    for lp in LOOSE_PRIMES:
        reg_keys = [f"{rt}/{rn}" for rt, rn in REGIONS]
        for i, rk in enumerate(reg_keys):
            rp2 = rprime_results[lp].get("Run 2", {}).get(rk, (float("nan"), float("nan"), 0))
            rp3 = rprime_results[lp].get("Run 3", {}).get(rk, (float("nan"), float("nan"), 0))
            sf2 = sf_results[lp].get("Run 2", {}).get(rk, (float("nan"), float("nan")))
            sf3 = sf_results[lp].get("Run 3", {}).get(rk, (float("nan"), float("nan")))
            lp_col  = LP_LABELS_TEX[lp] if i == 0 else ""
            reg_col = REGION_LABELS_TEX.get(rk, rk)
            lines.append(
                rf"{lp_col} & {reg_col}"
                rf" & {fmt_cell_tex(*rp2[:2])}"
                rf" & {fmt_cell_tex(*rp3[:2])}"
                rf" & {fmt_cell_tex(*sf2)}"
                rf" & {fmt_cell_tex(*sf3)} \\"
            )
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate R' and SF for ABCD background estimation.")
    parser.add_argument("--loose-prime", default="LoosePrime4",
                        choices=["LoosePrime4", "LoosePrime4a", "LoosePrime5", "Loose"],
                        help="LoosePrime working point to use (default: LoosePrime4)")
    parser.add_argument("--jfp-only", action="store_true",
                        help="Use only JFP (not JFP+Other) when computing R' (default: JFP+Other)")
    args = parser.parse_args()

    LOOSE_PRIMES[:] = [args.loose_prime]

    rprime_results = {}
    sf_results     = {}

    for lp in LOOSE_PRIMES:
        rprime_results[lp] = {}
        sf_results[lp]     = {}
        for era, run2 in [("Run 2", True), ("Run 3", False)]:
            rprime_results[lp][era] = collect_rprime_totals(lp, run2, jfp_only=args.jfp_only)
            sf_results[lp][era]     = collect_sf(lp, run2)

    # --- plain-text summary ---
    mc_label = "JFP" if args.jfp_only else "JFP+Other"
    col_rp, col_sf = f"R' ({mc_label})", "SF (data)"
    w = 22
    for lp in LOOSE_PRIMES:
        print(f"\n{'='*90}")
        print(f"  {lp}")
        print(f"{'='*90}")
        hdr = (f"  {'Region':<22}"
               f"  {'Run2 '+col_rp:>{w}}  {'Run3 '+col_rp:>{w}}"
               f"  {'Run2 '+col_sf:>{w}}  {'Run3 '+col_sf:>{w}}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for rt, rn in REGIONS:
            rk  = f"{rt}/{rn}"
            rp2 = rprime_results[lp]["Run 2"][rk]
            rp3 = rprime_results[lp]["Run 3"][rk]
            sf2 = sf_results[lp]["Run 2"][rk]
            sf3 = sf_results[lp]["Run 3"][rk]
            print(f"  {REGION_LABELS[rk]:<22}"
                  f"  {fmt_val(*rp2):>{w}}  {fmt_val(*rp3):>{w}}"
                  f"  {fmt_val(*sf2):>{w}}  {fmt_val(*sf3):>{w}}")

    # --- LaTeX table ---
    print("\n\n% ---- LaTeX table ----")
    print(make_latex_table(rprime_results, sf_results, jfp_only=args.jfp_only))

    # --- VR data vs prediction comparison ---
    print("\n\n% ---- VR data vs. prediction comparison ----")
    for run2, era in [(True, "Run 2"), (False, "Run 3")]:
        rp_era = rprime_results[args.loose_prime][era]
        print_vr_data_comparison(args.loose_prime, rp_era, run2, jfp_only=args.jfp_only)

    # --- Diagnostic: per-sample breakdown for VR-0L-mT-mid ---
    print("\n\n% ---- Per-sample diagnostic ----")
    for run2, era in [(True, "Run 2"), (False, "Run 3")]:
        print_sample_table(args.loose_prime, "VR", "0L-mT-mid", run2, jfp_only=args.jfp_only)
