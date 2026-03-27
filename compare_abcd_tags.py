#!/usr/bin/env python3
"""
Compare ABCD yields between two result directories ("tags") for a given MC
sample and the Run-2 data files, focused on VR-0L-mT-mid TT/TL/LT/LL yields.

Usage:
  python3 compare_abcd_tags.py DIR_A DIR_B [options]

Example:
  python3 compare_abcd_tags.py ABCD_results_4 ABCD_results_3.Y \\
      --sample Sh_2214_Znunu_pTV2_CVetoBVeto --loose-prime LoosePrime4
"""

import argparse
import glob
import json
import math
import os
import re

ABCD_BINS   = ("TT", "TL", "LT", "LL")
MC_CATS     = ("real", "efp", "jfp", "other", "unclassified")
RUN2_YEARS  = {2015, 2016, 2017, 2018}
RUN3_YEARS  = {2022, 2023, 2024}

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def parse_data_year(fname):
    """Extract 4-digit year from a data filename.
    Handles both old (data_2017) and new (data_17) formats."""
    m = re.search(r'_data_(\d{2,4})_', fname)
    if not m:
        return None
    y = int(m.group(1))
    return y + 2000 if y < 100 else y


def find_mc_file(results_dir, sample_substr, loose_prime):
    """Return the path(s) of MC files whose basename contains sample_substr."""
    pattern = os.path.join(results_dir,
                           f"*_ABCD_tightID_hybridCOIso_{loose_prime}.json")
    matches = [p for p in sorted(glob.glob(pattern))
               if sample_substr in os.path.basename(p)
               and "data_" not in os.path.basename(p)]
    return matches


def find_data_files(results_dir, loose_prime):
    """Return dict year -> path for all data files in the directory."""
    pattern = os.path.join(results_dir,
                           f"*_data_*_ABCD_tightID_hybridCOIso_{loose_prime}.json")
    result = {}
    for p in sorted(glob.glob(pattern)):
        fname = os.path.basename(p)
        year = parse_data_year(fname)
        if year is None:
            continue
        result[year] = p
    return result


def load_region(path, region):
    """Load one ABCD region dict from a JSON file.  Returns None if absent."""
    with open(path) as f:
        d = json.load(f)
    reg_type, reg_name = region.split("/", 1)
    try:
        return d[reg_type][reg_name]
    except KeyError:
        return None

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sw(bin_d, cat):
    """sumweights for a truth category in one bin, 0 if missing."""
    return bin_d.get(cat, {}).get("sumweights", 0.0)


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "     ---"
    return f"{v:>10.3f}"


def fmt_ratio(a, b):
    if b == 0:
        return "      inf" if a != 0 else "      ---"
    return f"{a/b:>9.4f}"


def header_line(w_label):
    return (f"  {'':>{w_label}}  {'Tag A':>10}  {'Tag B':>10}  {'A/B':>9}")


def sep_line(w_label):
    return "  " + "─" * (w_label + 34)

# ---------------------------------------------------------------------------
# Print routines
# ---------------------------------------------------------------------------

def print_mc_comparison(sample_substr, loose_prime, region, dir_a, dir_b,
                        label_a, label_b):
    files_a = find_mc_file(dir_a, sample_substr, loose_prime)
    files_b = find_mc_file(dir_b, sample_substr, loose_prime)

    if not files_a:
        print(f"  [MC] No file matching '{sample_substr}' in {label_a}")
        return
    if not files_b:
        print(f"  [MC] No file matching '{sample_substr}' in {label_b}")
        return

    # Match files by their bare sample name (prefix-stripped), so that
    # PICOPROD_RAv4_<sample>_ABCD_... and output_<sample>_ABCD_... pair correctly.
    def bare(path):
        fname = os.path.basename(path)
        for prefix in ("PICOPROD_RAv4_", "output_"):
            if fname.startswith(prefix):
                fname = fname[len(prefix):]
                break
        return fname

    keyed_a = {bare(p): p for p in files_a}
    keyed_b = {bare(p): p for p in files_b}
    common  = sorted(set(keyed_a) & set(keyed_b))
    only_a  = sorted(set(keyed_a) - set(keyed_b))
    only_b  = sorted(set(keyed_b) - set(keyed_a))

    pairs = [(k, keyed_a[k], keyed_b[k]) for k in common]
    for k in only_a:
        pairs.append((k, keyed_a[k], None))
    for k in only_b:
        pairs.append((k, None, keyed_b[k]))

    w = max(len(n) for n, _, _ in pairs)
    w = max(w, 20)

    for fname, path_a, path_b in sorted(pairs):
        print(f"\n  ── MC sample: {fname}")
        reg_a = load_region(path_a, region) if path_a else None
        reg_b = load_region(path_b, region) if path_b else None

        if reg_a is None and reg_b is None:
            print(f"    region '{region}' absent in both files")
            continue

        print(f"\n  {'':<30}  {'Tag A':>10}  {'Tag B':>10}  {'A/B':>9}")
        print(sep_line(30))

        for b in ABCD_BINS:
            ba = reg_a[b] if reg_a else None
            bb = reg_b[b] if reg_b else None
            # Total MC yield
            va = _sw(ba, "mc") if ba else float("nan")
            vb = _sw(bb, "mc") if bb else float("nan")
            print(f"  {b+' total MC':<30}  {fmt(va)}  {fmt(vb)}  {fmt_ratio(va, vb)}")
            # Per-category
            for cat in MC_CATS:
                ca = _sw(ba, cat) if ba else float("nan")
                cb = _sw(bb, cat) if bb else float("nan")
                if ca == 0 and cb == 0:
                    continue
                label = f"  └ {cat}"
                print(f"  {label:<30}  {fmt(ca)}  {fmt(cb)}  {fmt_ratio(ca, cb)}")
            print()


def print_data_comparison(loose_prime, region, dir_a, dir_b, label_a, label_b,
                          run2=True):
    era_years = RUN2_YEARS if run2 else RUN3_YEARS
    era_label = "Run 2" if run2 else "Run 3"

    files_a = {y: p for y, p in find_data_files(dir_a, loose_prime).items()
               if y in era_years}
    files_b = {y: p for y, p in find_data_files(dir_b, loose_prime).items()
               if y in era_years}

    all_years = sorted(set(files_a) | set(files_b))
    if not all_years:
        print(f"  No {era_label} data files found in either directory.")
        return

    # Per-year and running totals
    print(f"\n  ── Data ({era_label})")
    print(f"\n  {'Year / Bin':<20}  {'Tag A':>10}  {'Tag B':>10}  {'A/B':>9}")
    print(sep_line(20))

    total_a = {b: 0 for b in ABCD_BINS}
    total_b = {b: 0 for b in ABCD_BINS}

    for year in all_years:
        path_a = files_a.get(year)
        path_b = files_b.get(year)
        reg_a = load_region(path_a, region) if path_a else None
        reg_b = load_region(path_b, region) if path_b else None

        for b in ABCD_BINS:
            va = reg_a[b]["data"] if reg_a else None
            vb = reg_b[b]["data"] if reg_b else None
            label = f"{year} {b}"
            va_s = f"{va:>10d}" if va is not None else f"{'---':>10}"
            vb_s = f"{vb:>10d}" if vb is not None else f"{'---':>10}"
            ratio_s = fmt_ratio(va, vb) if (va is not None and vb is not None) else "      ---"
            print(f"  {label:<20}  {va_s}  {vb_s}  {ratio_s}")
            if va is not None: total_a[b] += va
            if vb is not None: total_b[b] += vb

        print()

    print(sep_line(20))
    for b in ABCD_BINS:
        label = f"TOTAL {b}"
        print(f"  {label:<20}  {total_a[b]:>10d}  {total_b[b]:>10d}  {fmt_ratio(total_a[b], total_b[b])}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ABCD yields between two result directories.")
    parser.add_argument("dir_a", metavar="DIR_A",
                        help="First results directory (Tag A)")
    parser.add_argument("dir_b", metavar="DIR_B",
                        help="Second results directory (Tag B)")
    parser.add_argument("--sample", default="Sh_2214_Znunu_pTV2_CVetoBVeto",
                        help="Substring to match MC sample filename "
                             "(default: Sh_2214_Znunu_pTV2_CVetoBVeto)")
    parser.add_argument("--loose-prime", default="LoosePrime4",
                        choices=["LoosePrime4", "LoosePrime4a", "LoosePrime5", "Loose"],
                        help="LoosePrime working point (default: LoosePrime4)")
    parser.add_argument("--region", default="VR/0L-mT-mid",
                        help="ABCD region to compare (default: VR/0L-mT-mid)")
    parser.add_argument("--run3", action="store_true",
                        help="Compare Run-3 data years instead of Run-2")
    args = parser.parse_args()

    label_a = os.path.basename(args.dir_a.rstrip("/"))
    label_b = os.path.basename(args.dir_b.rstrip("/"))

    print(f"\n{'='*70}")
    print(f"  Tag A : {args.dir_a}  ({label_a})")
    print(f"  Tag B : {args.dir_b}  ({label_b})")
    print(f"  Region: {args.region}   Working point: {args.loose_prime}")
    print(f"{'='*70}")

    # MC sample comparison
    print(f"\n{'─'*70}")
    print(f"  MC sample substring: '{args.sample}'")
    print(f"{'─'*70}")
    print_mc_comparison(args.sample, args.loose_prime, args.region,
                        args.dir_a, args.dir_b, label_a, label_b)

    # Data comparison
    print(f"\n{'─'*70}")
    run2 = not args.run3
    print_data_comparison(args.loose_prime, args.region,
                          args.dir_a, args.dir_b, label_a, label_b,
                          run2=run2)
