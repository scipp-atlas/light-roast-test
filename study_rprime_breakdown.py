#!/usr/bin/env python3
"""
Study R' = N_TT*N_LL / (N_TL*N_LT) broken down by:
  truth category : prompt, EFP, JFP, Other, JFP+Other
  process        : Znunu, Wtaunu, Wmunu, Wenu, nunugamma, inclusive
  era            : Run 2, Run 3
  ID working pt  : LoosePrime4, Loose
  photon pT bin  : pt < 25 GeV, pt > 25 GeV, inclusive
  region         : Preselection/0L, VR/0L-mT-mid, all SR regions

By default all regions and truth categories are printed.
Use --truth to restrict to a subset of truth categories.

R' and its uncertainty are computed from MC sumweights using the standard formula:
  staterr = sumweights / sqrt(nevents)
  delta_R' / R' = sqrt( (sigma_TT/N_TT)^2 + (sigma_TL/N_TL)^2 +
                        (sigma_LT/N_LT)^2 + (sigma_LL/N_LL)^2 )
"""

import argparse
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np
import uproot

from abcd_utils import get_region_masks, LoosePrimeDefs
from calc_rprime import calc_rprime, calc_rprime_err

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = "/data/mhance/SUSY/ntuples/v4.1"

TARGET_PROCESSES = ['Znunu', 'Wtaunu', 'Wmunu', 'Wenu', 'nunugamma']
ALL_PROCESSES    = TARGET_PROCESSES + ['inclusive']

TRUTH_CATS = ['prompt', 'EFP', 'JFP', 'Other', 'JFP+Other']
LP_WPS     = ['LoosePrime4', 'Loose']
ERAS       = ['Run2', 'Run3']

# pT bins: (label, lo_GeV, hi_GeV)  hi_GeV=None means no upper bound.
# Branch values are in MeV; multiply by 1e3 when building masks.
# Change entries here to adjust binning — no other edits needed.
PT_BINS = [
    ('10-15',  10.,  15.),
    ('15-25',  15.,  25.),
    ('25-40',  25.,  40.),
    ('40-80',  40.,  80.),
    ('>80',    80.,  None),
    ('incl',    0.,  None),
]

# Eta bins: (label, lo_abseta, hi_abseta)  None bounds mean no cut.
ETA_BINS = [
    ('barrel',  0.,    1.37),
    ('endcap',  1.52,  2.37),
    ('incl',    None,  None),
]

# Conversion bins: (label, converted)  converted=None means no cut (inclusive).
CONV_BINS = [
    ('unconverted', False),
    ('converted',   True),
    ('incl',        None),
]

# Derived label lists used for printing and CLI choices
PT_BIN_LABELS   = [label for label, _, _ in PT_BINS]
ETA_BIN_LABELS  = [label for label, _, _ in ETA_BINS]
CONV_BIN_LABELS = [label for label, _ in CONV_BINS]

DEFAULT_REGIONS = [
    ('Preselection', '0L'),
    ('VR',           '0L-mT-mid'),
    ('SR',           '0L-mT-low-loose'),
    ('SR',           '0L-mT-mid-loose'),
    ('SR',           '0L-mT-hgh-loose'),
]

NEEDED_BRANCHES = [
    'ph_pt', 'ph_eta', 'ph_conversionType', 'ph_truthprompt', 'ph_truthEFP', 'ph_truthJFP', 'ph_truthother',
    'ph_select_baseline', 'ph_select_tightID', 'ph_isEM', 'ph_select_hybridCOIso',
    'weight_total', 'weight_fjvt_effSF', 'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
    'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nPhotons_baseline', 'nPhotons_skims', 'nPhotons_baseline_noOR',
    'mTGammaMet', 'dPhiGammaMet', 'nTau20_veryloose', 'met_signif',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_file(fname):
    """Return (process, era) or None."""
    if   'mc20' in fname: era = 'Run2'
    elif 'mc23' in fname: era = 'Run3'
    else: return None
    for proc in TARGET_PROCESSES:
        if proc in fname:
            return proc, era
    return None


def rp_val(bins):
    """Compute R' and uncertainty from {'TT':{'sw':..,'sw2':..}, ...}."""
    sw = {b: bins[b]['sw']  for b in ('TT','TL','LT','LL')}
    se = {b: math.sqrt(bins[b]['sw2']) for b in ('TT','TL','LT','LL')}
    if any(sw[b] <= 0 for b in ('TT','TL','LT','LL')):
        return float('nan'), float('nan')
    rp  = calc_rprime(sw['TT'], sw['TL'], sw['LT'], sw['LL'])
    err = calc_rprime_err(rp,
                          sw['TT'], se['TT'], sw['TL'], se['TL'],
                          sw['LT'], se['LT'], sw['LL'], se['LL'])
    return rp, err


def rp_str(bins):
    rp, err = rp_val(bins)
    if math.isnan(rp):
        return '      ---      '
    return f'{rp:6.3f} ± {err:5.3f}'

# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(regions):
    """
    Loop over all v4.1 ntuples and accumulate yields per
    (region, era, lp, process, truth_cat, conv_bin, eta_bin, pt_bin, abcd_bin).

    regions — list of (region_type, region_name) pairs

    Returns a nested defaultdict:
      acc[region_key][era][lp][proc][truth][conv_bin][eta_bin][pt_bin][abcd_bin] = {'sw': float, 'n': int}
    where region_key = f"{region_type}/{region_name}"
    """
    def _leaf():
        return {'sw': 0.0, 'sw2': 0.0, 'n': 0}

    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(
          lambda: defaultdict(lambda: defaultdict(
          lambda: defaultdict(lambda: defaultdict(
          lambda: defaultdict(lambda: defaultdict(_leaf)))))))))

    files = []
    for fp in sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root'))):
        info = classify_file(os.path.basename(fp))
        if info is not None:
            files.append((fp, *info))

    region_labels = ', '.join(f'{rt}/{rn}' for rt, rn in regions)
    print(f"Reading {len(files)} files for regions: {region_labels} ...")

    for fp, proc, era in files:
        print(f"  {era}  {proc:<12}  {os.path.basename(fp)}")
        try:
            with uproot.open(fp) as uf:
                if 'picontuple' not in uf:
                    print("    (no picontuple, skipping)")
                    continue
                data = uf['picontuple'].arrays(NEEDED_BRANCHES, library='np')
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        totalweight = (
            data['weight_total'] *
            data['weight_fjvt_effSF'] *
            data['weight_ftag_effSF_GN2v01_Continuous'] *
            data['weight_jvt_effSF']
        )

        all_region_masks = get_region_masks(data)

        baseline = data['ph_select_baseline'] == 1
        tight    = data['ph_select_tightID']  == 1
        iso      = data['ph_select_hybridCOIso'] == 1

        truth_masks = {
            'prompt':    data['ph_truthprompt'] == 1,
            'EFP':       data['ph_truthEFP']    == 1,
            'JFP':       data['ph_truthJFP']    == 1,
            'Other':     data['ph_truthother']  == 1,
            'JFP+Other': (data['ph_truthJFP'] == 1) | (data['ph_truthother'] == 1),
        }

        pt       = data['ph_pt']   # MeV
        aeta     = np.abs(data['ph_eta'])
        is_conv  = data['ph_conversionType'] > 0

        pt_masks = {
            label: (pt >= lo * 1e3) & (pt < hi * 1e3 if hi is not None else True)
            for label, lo, hi in PT_BINS
        }
        eta_masks = {
            label: (
                np.ones(len(aeta), dtype=bool) if (lo is None and hi is None)
                else (aeta >= lo) & (aeta < hi)
            )
            for label, lo, hi in ETA_BINS
        }
        conv_masks = {
            label: (
                np.ones(len(is_conv), dtype=bool) if converted is None
                else (is_conv if converted else ~is_conv)
            )
            for label, converted in CONV_BINS
        }

        for lp in LP_WPS:
            lp_bit   = LoosePrimeDefs[lp]
            lp_shape = (data['ph_isEM'] & lp_bit) == 0

            abcd_masks = {
                'TT': baseline &  tight &  iso & lp_shape,
                'TL': baseline &  tight & ~iso & lp_shape,
                'LT': baseline & ~tight &  iso & lp_shape,
                'LL': baseline & ~tight & ~iso & lp_shape,
            }

            for rt, rn in regions:
                reg_key  = f"{rt}/{rn}"
                reg_mask = all_region_masks[rt][rn]
                if not reg_mask.any():
                    continue

                for truth, tmask in truth_masks.items():
                    for conv_label, cmask in conv_masks.items():
                        for eta_label, emask in eta_masks.items():
                            for pt_label, pmask in pt_masks.items():
                                base = reg_mask & tmask & cmask & emask & pmask
                                for b, amask in abcd_masks.items():
                                    sel = base & amask
                                    if not sel.any():
                                        continue
                                    w   = totalweight[sel]
                                    sw  = float(w.sum())
                                    sw2 = float((w ** 2).sum())
                                    n   = int(sel.sum())
                                    for bucket in (
                                        acc[reg_key][era][lp][proc][truth][conv_label][eta_label][pt_label][b],
                                        acc[reg_key][era][lp]['inclusive'][truth][conv_label][eta_label][pt_label][b],
                                    ):
                                        bucket['sw']  += sw
                                        bucket['sw2'] += sw2
                                        bucket['n']   += n

    return acc

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_tables(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins):
    col_w   = 17   # width of each R' ± err cell
    proc_w  = 12
    truth_w = 11

    header_cols = '  '.join(f'{label:^{col_w}}' for label, _, _ in PT_BINS)
    header = f'{"Truth":<{truth_w}}  {"Process":<{proc_w}}  {header_cols}'
    sep    = '-' * len(header)

    for rt, rn in regions:
        reg_key = f"{rt}/{rn}"
        reg_acc = acc[reg_key]
        for era in ERAS:
            for lp in wps:
                for conv_label, converted in conv_bins:
                    for eta_label, eta_lo, eta_hi in eta_bins:
                        if eta_lo is None:
                            eta_desc = 'eta: inclusive'
                        else:
                            eta_desc = f'|eta| in [{eta_lo}, {eta_hi})'
                        print(f'\n{"="*len(header)}')
                        print(f'{era}  |  {lp}  |  {reg_key}  |  {conv_label}  |  {eta_desc}')
                        print('='*len(header))
                        print(header)

                        prev_truth = None
                        for truth in truth_cats:
                            if prev_truth is not None:
                                print()
                            prev_truth = truth
                            print(sep)
                            for proc in processes:
                                cells = '  '.join(
                                    f'{rp_str(reg_acc[era][lp][proc][truth][conv_label][eta_label][pt_label]):^{col_w}}'
                                    for pt_label, _, _ in PT_BINS
                                )
                                print(f'{truth:<{truth_w}}  {proc:<{proc_w}}  {cells}')

                        print(sep)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="R' breakdown by truth category, process, pT, WP, era, and region.")
    parser.add_argument(
        '--truth', nargs='+', metavar='CAT',
        choices=TRUTH_CATS, default=None,
        help=('Truth categories to print (default: all). '
              f'Choices: {", ".join(TRUTH_CATS)}'))
    parser.add_argument(
        '--regions', nargs='+', metavar='TYPE/NAME',
        default=None,
        help=('Regions to process, as TYPE/NAME pairs e.g. Preselection/0L '
              '(default: all regions)'))
    parser.add_argument(
        '--wp', nargs='+', metavar='WP',
        choices=LP_WPS, default=None,
        help=('ID working points to print (default: all). '
              f'Choices: {", ".join(LP_WPS)}'))
    parser.add_argument(
        '--processes', nargs='+', metavar='PROC',
        choices=ALL_PROCESSES, default=None,
        help=('Processes to print (default: all). '
              f'Choices: {", ".join(ALL_PROCESSES)}'))
    parser.add_argument(
        '--eta', nargs='+', metavar='ETA',
        choices=ETA_BIN_LABELS, default=None,
        help=('Eta bins to print (default: all). '
              f'Choices: {", ".join(ETA_BIN_LABELS)}'))
    parser.add_argument(
        '--conv', nargs='+', metavar='CONV',
        choices=CONV_BIN_LABELS, default=None,
        help=('Conversion categories to print (default: all). '
              f'Choices: {", ".join(CONV_BIN_LABELS)}'))
    args = parser.parse_args()

    truth_cats = args.truth      if args.truth      else TRUTH_CATS
    wps        = args.wp         if args.wp         else LP_WPS
    processes  = args.processes  if args.processes  else ALL_PROCESSES
    eta_bins   = [(l, lo, hi) for l, lo, hi in ETA_BINS  if l in (args.eta  or ETA_BIN_LABELS)]
    conv_bins  = [(l, c)      for l, c      in CONV_BINS if l in (args.conv or CONV_BIN_LABELS)]

    if args.regions:
        regions = []
        for r in args.regions:
            parts = r.split('/', 1)
            if len(parts) != 2:
                parser.error(f"Region must be TYPE/NAME, got: {r!r}")
            regions.append((parts[0], parts[1]))
    else:
        regions = DEFAULT_REGIONS

    acc = accumulate(regions)
    print_tables(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins)
