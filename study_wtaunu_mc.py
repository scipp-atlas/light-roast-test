#!/usr/bin/env python3
"""
Study whether treating Wtaunu JFP+Other as an MC-estimated background (rather
than data-driven) changes the ABCD closure.

Two predictions are compared for data in each region:

  Standard ABCD:
    sidebands = data - MC(prompt+EFP)
    pred_TT   = MC_prompt_TT + MC_EFP_TT + R'_all * ABCD_all(sidebands)

  Modified ABCD  (Wtaunu JFP from MC):
    sidebands = data - MC(prompt+EFP) - MC_Wtaunu(JFP+Other)
    pred_TT   = MC_prompt_TT + MC_EFP_TT + MC_Wtaunu_JFP_TT
              + R'_noWtaunu * ABCD_noWtaunu(sidebands)

R'_all      is computed from all JFP+Other MC (same as calc_rprime.py).
R'_noWtaunu is computed from non-Wtaunu JFP+Other MC only.

Scale factors are SF = data_TT / pred_TT for each method.
"""

import argparse
import glob
import json
import math
import os

from calc_rprime import (
    RESULTS_DIR, REGIONS, SKIP_TAGS, RUN2_YEARS, RUN3_YEARS,
    SR_MT_KEYS, SR_MT_LOOSE_KEYS, SR_COMBINED, SR_COMBINED_LOOSE,
    REGION_LABELS,
    is_run2_mc, is_run3_mc, parse_data_year, sample_name,
    jfp_other, prompt, calc_rprime, calc_rprime_err,
)

WTAUNU_SUBSTR = "Wtaunu"

# Regions available for the data-vs-prediction comparison table
COMPARE_REGIONS = {
    "VR/0L-mT-mid":    ("VR",  "0L-mT-mid"),
    "SR/0L-loose":     None,   # combined — handled specially
    "SR/0L-mT-low-loose":  ("SR", "0L-mT-low-loose"),
    "SR/0L-mT-mid-loose":  ("SR", "0L-mT-mid-loose"),
    "SR/0L-mT-hgh-loose":  ("SR", "0L-mT-hgh-loose"),
}

# ---------------------------------------------------------------------------
# Accumulate yields
# ---------------------------------------------------------------------------

def _zero_bins():
    return {b: 0.0 for b in ("TT", "TL", "LT", "LL")}


def collect_yields(loose_prime, run2):
    """
    Read all JSON files for the given era and accumulate, per ABCD region:

      data_n          : raw data event counts
      prompt_sw/e2    : MC prompt+EFP subtraction (all MC, all samples)
      wtaunu_sw/e2    : Wtaunu JFP+Other yields from MC
      jfp_all_sw/e2   : all JFP+Other MC yields  (for R'_all)
      jfp_nowt_sw/e2  : non-Wtaunu JFP+Other MC  (for R'_noWtaunu)
      mc_tt_*         : per-category TT yields (for full-MC prediction)

    Also accumulates combined SR keys (SR/0L, SR/0L-loose).
    """
    all_region_keys = [f"{rt}/{rn}" for rt, rn in REGIONS] + \
                      [SR_COMBINED, SR_COMBINED_LOOSE]

    data_n        = {k: {b: 0.0 for b in ("TT","TL","LT","LL")} for k in all_region_keys}
    prompt_sw     = {k: _zero_bins() for k in all_region_keys}
    prompt_e2     = {k: _zero_bins() for k in all_region_keys}
    wtaunu_sw     = {k: _zero_bins() for k in all_region_keys}
    wtaunu_e2     = {k: _zero_bins() for k in all_region_keys}
    jfp_all_sw    = {k: _zero_bins() for k in all_region_keys}
    jfp_all_e2    = {k: _zero_bins() for k in all_region_keys}
    jfp_nowt_sw   = {k: _zero_bins() for k in all_region_keys}
    jfp_nowt_e2   = {k: _zero_bins() for k in all_region_keys}
    mc_tt_prompt  = {k: (0.0, 0.0) for k in all_region_keys}  # (sw, se2)
    mc_tt_efp     = {k: (0.0, 0.0) for k in all_region_keys}
    mc_tt_wtaunu  = {k: (0.0, 0.0) for k in all_region_keys}
    mc_tt_total   = {k: (0.0, 0.0) for k in all_region_keys}

    pattern = os.path.join(RESULTS_DIR,
                           f"*_ABCD_tightID_hybridCOIso_{loose_prime}.json")

    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)

        # ---- data files ----
        if "data_" in fname:
            year = parse_data_year(fname)
            if year is None:
                continue
            era_years = RUN2_YEARS if run2 else RUN3_YEARS
            if year not in era_years:
                continue
            with open(path) as f:
                d = json.load(f)
            for reg_type, reg_name in REGIONS:
                key = f"{reg_type}/{reg_name}"
                is_sr = key.startswith("SR/")
                try:
                    reg = d[reg_type][reg_name]
                except KeyError:
                    continue
                for b in ("TT", "TL", "LT", "LL"):
                    if is_sr and b == "TT":
                        continue  # blind SR TT — no data
                    _d = reg[b]["data"]
                    _sw = _d["sumweights"] if isinstance(_d, dict) else float(_d)
                    data_n[key][b] += _sw
                    if key in SR_MT_KEYS and b != "TT":
                        data_n[SR_COMBINED][b] += _sw
                    if key in SR_MT_LOOSE_KEYS and b != "TT":
                        data_n[SR_COMBINED_LOOSE][b] += _sw
            continue

        # ---- background MC files ----
        if any(t in fname for t in SKIP_TAGS):
            continue
        if run2 and not is_run2_mc(path):   continue
        if not run2 and not is_run3_mc(path): continue

        is_wt = WTAUNU_SUBSTR in fname

        with open(path) as f:
            d = json.load(f)

        for reg_type, reg_name in REGIONS:
            key = f"{reg_type}/{reg_name}"
            try:
                reg = d[reg_type][reg_name]
            except KeyError:
                continue

            combined_keys = []
            if key in SR_MT_KEYS:       combined_keys.append(SR_COMBINED)
            if key in SR_MT_LOOSE_KEYS: combined_keys.append(SR_COMBINED_LOOSE)
            affected = [key] + combined_keys

            for b in ("TT", "TL", "LT", "LL"):
                bin_d = reg[b]

                # prompt subtraction (all MC)
                p_sw, p_se = prompt(bin_d)
                # JFP+Other
                j_sw, j_se = jfp_other(bin_d)

                for k in affected:
                    prompt_sw[k][b]  += p_sw
                    prompt_e2[k][b]  += p_se**2
                    jfp_all_sw[k][b] += j_sw
                    jfp_all_e2[k][b] += j_se**2

                    if is_wt:
                        wtaunu_sw[k][b]  += j_sw
                        wtaunu_e2[k][b]  += j_se**2
                    else:
                        jfp_nowt_sw[k][b] += j_sw
                        jfp_nowt_e2[k][b] += j_se**2

            # TT per-category breakdown
            bin_tt = reg["TT"]
            p_sw_tt, p_se_tt = prompt(bin_tt)
            efp_sw = bin_tt["efp"]["sumweights"]
            efp_se = bin_tt["efp"]["staterr"]
            # prompt for TT display = real + unclassified (no EFP)
            real_sw = bin_tt["real"]["sumweights"] + bin_tt["unclassified"]["sumweights"]
            real_se2 = bin_tt["real"]["staterr"]**2 + bin_tt["unclassified"]["staterr"]**2
            j_sw_tt, j_se_tt = jfp_other(bin_tt)
            total_sw = sum(bin_tt[c]["sumweights"] for c in
                           ("real","efp","jfp","other","unclassified"))
            total_se2 = sum(bin_tt[c]["staterr"]**2 for c in
                            ("real","efp","jfp","other","unclassified"))

            for k in affected:
                sw0, se0 = mc_tt_prompt[k]
                mc_tt_prompt[k] = (sw0 + real_sw, se0 + real_se2)
                sw0, se0 = mc_tt_efp[k]
                mc_tt_efp[k]    = (sw0 + efp_sw, se0 + efp_se**2)
                if is_wt:
                    sw0, se0 = mc_tt_wtaunu[k]
                    mc_tt_wtaunu[k] = (sw0 + j_sw_tt, se0 + j_se_tt**2)
                sw0, se0 = mc_tt_total[k]
                mc_tt_total[k]  = (sw0 + total_sw, se0 + total_se2)

    return dict(
        data_n=data_n,
        prompt_sw=prompt_sw, prompt_e2=prompt_e2,
        wtaunu_sw=wtaunu_sw, wtaunu_e2=wtaunu_e2,
        jfp_all_sw=jfp_all_sw, jfp_all_e2=jfp_all_e2,
        jfp_nowt_sw=jfp_nowt_sw, jfp_nowt_e2=jfp_nowt_e2,
        mc_tt_prompt=mc_tt_prompt,
        mc_tt_efp=mc_tt_efp,
        mc_tt_wtaunu=mc_tt_wtaunu,
        mc_tt_total=mc_tt_total,
    )


def rprime_from_totals(sw, e2):
    """Compute R' and uncertainty from {TT,TL,LT,LL} sumweight / variance dicts."""
    n  = sw
    e2d = e2
    rp = calc_rprime(n["TT"], n["TL"], n["LT"], n["LL"])
    if math.isnan(rp):
        return float("nan"), float("nan")
    sigma = calc_rprime_err(rp,
        n["TT"], math.sqrt(e2d["TT"]),
        n["TL"], math.sqrt(e2d["TL"]),
        n["LT"], math.sqrt(e2d["LT"]),
        n["LL"], math.sqrt(e2d["LL"]))
    return rp, sigma


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def abcd_estimate(rp, rp_err, fake, fake_e2):
    """R'-scaled ABCD estimate and uncertainty for a given set of fake sideband yields."""
    if any(fake[b] <= 0 for b in ("TL","LT","LL")):
        return float("nan"), float("nan")
    dd = fake["TL"] * fake["LT"] / fake["LL"]
    dd_err = dd * math.sqrt(
        fake_e2["TL"] / fake["TL"]**2 +
        fake_e2["LT"] / fake["LT"]**2 +
        fake_e2["LL"] / fake["LL"]**2
    )
    if math.isnan(rp) or rp <= 0:
        return float("nan"), float("nan")
    est     = rp * dd
    est_err = est * math.sqrt((rp_err / rp)**2 + (dd_err / dd)**2)
    return est, est_err


def print_region_comparison(key, yields, run2):
    era = "Run 2" if run2 else "Run 3"
    label = REGION_LABELS.get(key, key)
    print(f"\n  {'═'*72}")
    print(f"  {label}  |  {era}")
    print(f"  {'═'*72}")

    data_n  = yields["data_n"][key]
    pmt_sw  = yields["prompt_sw"][key]
    pmt_e2  = yields["prompt_e2"][key]
    wt_sw   = yields["wtaunu_sw"][key]
    wt_e2   = yields["wtaunu_e2"][key]
    all_sw  = yields["jfp_all_sw"][key]
    all_e2  = yields["jfp_all_e2"][key]
    nowt_sw = yields["jfp_nowt_sw"][key]
    nowt_e2 = yields["jfp_nowt_e2"][key]

    # R' values
    rp_all,  rp_all_err  = rprime_from_totals(all_sw,  all_e2)
    rp_nowt, rp_nowt_err = rprime_from_totals(nowt_sw, nowt_e2)

    # Sideband fake yields — standard (data - prompt - EFP)
    fake_std    = {b: data_n[b] - pmt_sw[b]            for b in ("TL","LT","LL")}
    fake_std_e2 = {b: data_n[b] + pmt_e2[b]            for b in ("TL","LT","LL")}

    # Sideband fake yields — modified (also subtract Wtaunu JFP+Other)
    fake_mod    = {b: fake_std[b] - wt_sw[b]            for b in ("TL","LT","LL")}
    fake_mod_e2 = {b: fake_std_e2[b] + wt_e2[b]        for b in ("TL","LT","LL")}

    # MC TT quantities
    mc_prompt_sw,  mc_prompt_e2  = yields["mc_tt_prompt"][key]
    mc_efp_sw,     mc_efp_e2     = yields["mc_tt_efp"][key]
    mc_wtaunu_sw,  mc_wtaunu_e2  = yields["mc_tt_wtaunu"][key]
    mc_total_sw,   mc_total_e2   = yields["mc_tt_total"][key]
    mc_prompt_e = math.sqrt(mc_prompt_e2)
    mc_efp_e    = math.sqrt(mc_efp_e2)
    mc_wtaunu_e = math.sqrt(mc_wtaunu_e2)
    mc_total_e  = math.sqrt(mc_total_e2)

    # ABCD estimates
    abcd_std,  abcd_std_err  = abcd_estimate(rp_all,  rp_all_err,  fake_std, fake_std_e2)
    abcd_mod,  abcd_mod_err  = abcd_estimate(rp_nowt, rp_nowt_err, fake_mod, fake_mod_e2)

    # Hybrid predictions
    def hybrid(mc_a, mc_a_e, mc_b, mc_b_e, abcd, abcd_e):
        tot = mc_a + mc_b + abcd
        err = math.sqrt(mc_a_e**2 + mc_b_e**2 + abcd_e**2) if not math.isnan(abcd) else float("nan")
        return tot, err

    pred_std, pred_std_err = hybrid(mc_prompt_sw, mc_prompt_e,
                                     mc_efp_sw,    mc_efp_e,
                                     abcd_std,     abcd_std_err)
    pred_mod, pred_mod_err = hybrid(mc_prompt_sw + mc_efp_sw + mc_wtaunu_sw,
                                     math.sqrt(mc_prompt_e2 + mc_efp_e2 + mc_wtaunu_e2),
                                     0.0, 0.0,
                                     abcd_mod, abcd_mod_err)

    is_sr = key.startswith("SR/")

    data_tt   = data_n["TT"]
    data_tt_e = math.sqrt(data_tt) if data_tt > 0 else 0.0

    # Scale factors (only for non-SR regions)
    def sf(data, pred, pred_e):
        if math.isnan(pred) or pred <= 0:
            return float("nan"), float("nan")
        s = data / pred
        s_e = s * math.sqrt((data_tt_e / data)**2 + (pred_e / pred)**2)
        return s, s_e

    if not is_sr:
        sf_std, sf_std_e = sf(data_tt, pred_std, pred_std_err)
        sf_mod, sf_mod_e = sf(data_tt, pred_mod, pred_mod_err)

    # MC JFP+Other scale factor (ABCD_std / MC_JFP+Other in TT)
    mc_jfp_other_tt = mc_total_sw - mc_prompt_sw - mc_efp_sw
    mc_jfp_other_e2 = mc_total_e2 + mc_prompt_e2 + mc_efp_e2
    if not math.isnan(abcd_std) and mc_jfp_other_tt > 0:
        sf_jfp_mc = abcd_std / mc_jfp_other_tt
        sf_jfp_mc_e = sf_jfp_mc * math.sqrt(
            (abcd_std_err / abcd_std)**2 + mc_jfp_other_e2 / mc_jfp_other_tt**2)
    else:
        sf_jfp_mc, sf_jfp_mc_e = float("nan"), float("nan")

    # --- Printing ---
    w = 10
    def row(label, val, err=None, indent=4):
        pfx = " " * indent
        val_s = f"{val:>{w}.2f}" if not math.isnan(val) else f"{'---':>{w}}"
        err_s = f"±{err:>{w-1}.2f}" if (err is not None and not math.isnan(err)) else f"{'':>{w+1}}"
        print(f"{pfx}{label:<46}  {val_s}  {err_s}")

    def section(title):
        print(f"\n  {title}")
        print(f"  {'─'*72}")

    # R' summary
    rp_tag_all  = f"R'_all={rp_all:.4f}±{rp_all_err:.4f}"    if not math.isnan(rp_all)  else "R'_all=---"
    rp_tag_nowt = f"R'_noWt={rp_nowt:.4f}±{rp_nowt_err:.4f}" if not math.isnan(rp_nowt) else "R'_noWt=---"
    print(f"\n    {rp_tag_all}    {rp_tag_nowt}")

    section("Sideband fake yields")
    for b in ("TL","LT","LL"):
        e_std = math.sqrt(fake_std_e2[b])
        e_mod = math.sqrt(fake_mod_e2[b])
        print(f"    fake {b}  std: {fake_std[b]:>10.2f} ±{e_std:>8.2f}   "
              f"mod: {fake_mod[b]:>10.2f} ±{e_mod:>8.2f}")

    if not is_sr:
        section("(a) Data TT")
        row("Data TT (observed)", data_tt, data_tt_e)

    section("(b) Full MC TT")
    row("Total MC TT",                   mc_total_sw,    mc_total_e)
    row("  prompt (real+unclassified)",   mc_prompt_sw,   mc_prompt_e)
    row("  EFP",                          mc_efp_sw,      mc_efp_e)
    row("  JFP+Other (non-Wtaunu)",       mc_jfp_other_tt - mc_wtaunu_sw,
                                          math.sqrt(mc_jfp_other_e2 + mc_wtaunu_e2))
    row("  JFP+Other (Wtaunu)",           mc_wtaunu_sw,   mc_wtaunu_e)

    section(f"(c) Standard ABCD [{rp_tag_all}]")
    row("MC prompt TT",                   mc_prompt_sw,  mc_prompt_e)
    row("MC EFP TT",                      mc_efp_sw,     mc_efp_e)
    row("ABCD JFP+Other (R'-scaled)",     abcd_std,      abcd_std_err)
    row("Total",                          pred_std,      pred_std_err)
    if not is_sr:
        row("SF  data / standard",            sf_std,        sf_std_e)
        row("SF  ABCD(JFP+Other) / MC(JFP+Other)",   sf_jfp_mc,  sf_jfp_mc_e)

    section(f"(d) Modified ABCD — Wtaunu from MC [{rp_tag_nowt}]")
    row("MC prompt TT",                   mc_prompt_sw,  mc_prompt_e)
    row("MC EFP TT",                      mc_efp_sw,     mc_efp_e)
    row("MC Wtaunu JFP+Other TT",         mc_wtaunu_sw,  mc_wtaunu_e)
    row("ABCD non-Wtaunu (R'-scaled)",    abcd_mod,      abcd_mod_err)
    row("Total",                          pred_mod,      pred_mod_err)
    if not is_sr:
        row("SF  data / modified",            sf_mod,        sf_mod_e)
    print(f"  {'─'*72}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare standard vs. Wtaunu-from-MC ABCD estimates.")
    parser.add_argument("--loose-prime", default="LoosePrime4",
                        choices=["LoosePrime4", "LoosePrime4a", "LoosePrime5", "Loose"],
                        help="LoosePrime working point (default: LoosePrime4)")
    parser.add_argument("--region", default=None,
                        choices=list(COMPARE_REGIONS.keys()),
                        help="Region for data vs. prediction comparison "
                             "(default: all available)")
    parser.add_argument("--run3", action="store_true",
                        help="Use Run-3 data/MC instead of Run-2")
    args = parser.parse_args()

    run2 = not args.run3
    era  = "Run 2" if run2 else "Run 3"

    regions_to_show = ([args.region] if args.region
                       else list(COMPARE_REGIONS.keys()))

    print(f"\nWtaunu-MC study  |  {args.loose_prime}  |  {era}")
    print(f"Results from: {RESULTS_DIR}")

    yields = collect_yields(args.loose_prime, run2)

    for key in regions_to_show:
        # Handle combined SR/0L-loose (sum of three mT bins — already in yields)
        print_region_comparison(key, yields, run2)
