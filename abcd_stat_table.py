#!/usr/bin/env python3
"""
Print a table of ABCD background estimates and statistical uncertainties
for each LooseID working point, using data files from ABCD_results_4.1.

Files matched: *_data_<year>_ABCD_tightID_hybridCOIso_<LooseID>.json
ABCD estimate: TT_est = TL * LT / LL
Stat error:    delta(TT_est) = TT_est * sqrt(1/TL + 1/LT + 1/LL)
"""

import glob
import json
import math
import os
import re
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "ABCD_results_4.1")

LOOSE_ID_ORDER = ["Loose", "LoosePrime4", "LoosePrime5", "LoosePrime4a"]

SR_REGION_ORDER = ["SR/0L-mT-low", "SR/0L-mT-mid", "SR/0L-mT-hgh", "VR/0L-mT-mid"]

RUN_ERAS = {
    "Run 2": lambda year: year < 2020,
    "Run 3": lambda year: year > 2020,
}


def load_counts(results_dir):
    """Sum data counts per era for each (era, LooseID, region, subregion, ABCD key)."""
    counts = {
        era: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
        for era in RUN_ERAS
    }

    pattern = os.path.join(results_dir, "*_data_*_ABCD_tightID_hybridCOIso_*.json")
    for path in glob.glob(pattern):
        basename = os.path.basename(path)
        m = re.search(r'_data_(\d{2,4})_', basename)
        if not m:
            continue
        y = int(m.group(1))
        year = y + 2000 if y < 100 else y
        loose_id = basename.split("hybridCOIso_")[1].replace(".json", "")

        era = next((e for e, fn in RUN_ERAS.items() if fn(year)), None)
        if era is None:
            continue

        with open(path) as f:
            data = json.load(f)
        for region, subregions in data.items():
            for subregion, abcd in subregions.items():
                for key in ("TT", "TL", "LT", "LL"):
                    if key in abcd:
                        val = abcd[key]
                        counts[era][loose_id][region][subregion][key] += (
                            val["data"] if isinstance(val, dict) else val
                        )
    return counts


def abcd_estimate(tl, lt, ll):
    """Return (TT_est, abs_err, rel_err) or (None, None, None) if inputs are invalid."""
    if tl <= 0 or lt <= 0 or ll <= 0:
        return None, None, None
    tt_est = tl * lt / ll
    rel_err = math.sqrt(1 / tl + 1 / lt + 1 / ll)
    return tt_est, tt_est * rel_err, rel_err


def fmt(val, fmt_spec):
    return f"{val:{fmt_spec}}" if val is not None else "—"


def print_sr_table(counts):
    header = f"{'Region':<18} {'LooseID':<14} {'TL':>6} {'LT':>6} {'LL':>6}  {'TT_est':>8}  {'± abs':>8}  {'± rel':>7}"
    sep = "-" * len(header)

    for era, era_counts in counts.items():
        loose_ids = [lid for lid in LOOSE_ID_ORDER if lid in era_counts]
        print(f"\n{'=' * len(header)}")
        print(f"{era}")
        print(header)

        for i, region_full in enumerate(SR_REGION_ORDER):
            if i > 0:
                print()
            region, subregion = region_full.split("/", 1)
            print(sep)
            for lid in loose_ids:
                r = era_counts[lid][region][subregion]
                tl, lt, ll = r["TL"], r["LT"], r["LL"]
                tt_est, abs_err, rel_err = abcd_estimate(tl, lt, ll)

                rel_str = f"{rel_err * 100:.1f}%" if rel_err is not None else "∞"
                abs_str = f"±{abs_err:.2f}" if abs_err is not None else "—"
                tt_str  = f"{tt_est:.2f}"   if tt_est  is not None else "—"

                print(f"{region_full:<18} {lid:<14} {tl:>6} {lt:>6} {ll:>6}  {tt_str:>8}  {abs_str:>8}  {rel_str:>7}")

        print(sep)


if __name__ == "__main__":
    counts = load_counts(RESULTS_DIR)
    print_sr_table(counts)
