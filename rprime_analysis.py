#!/usr/bin/env python3
"""
Unified R' analysis tool: summary plots and/or tabular output.

Plots (default): R' vs pT, eta, conversion status, process, and region.
Tables (--fulltables): full breakdown by truth category, process, pT, WP, era, region.

Use --no-plots to skip plot generation.
Use --latex for LaTeX-formatted table output.

Output: rprime_analysis_output/rprime_vs_<variable>_<truth>_<era_tag>.{pdf,png,tex}
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
if 'Nimbus Sans' in {f.name for f in _fm.fontManager.ttflist}:
    matplotlib.rcParams['font.family'] = 'Nimbus Sans'
import numpy as np
import uproot

from abcd_utils import get_region_masks, LoosePrimeDefs
from calc_rprime import calc_rprime, calc_rprime_err

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = "/data/mhance/SUSY/ntuples/v4.2"

TARGET_PROCESSES = ['Znunu', 'Wtaunu', 'Wmunu', 'Wenu', 'nunugamma']
ALL_PROCESSES    = TARGET_PROCESSES + ['inclusive']

TRUTH_CATS = ['prompt', 'EFP', 'JFP', 'Other', 'JFP+Other']
LP_WPS     = ['LoosePrime4', 'Loose']
ERAS       = ['Run2', 'Run3']

# pT bins: (label, lo_GeV, hi_GeV)  hi_GeV=None means no upper bound.
PT_BINS = [
    ('10-15',  10.,  15.),
    ('15-20',  15.,  20.),
    ('20-25',  20.,  25.),
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

# Conversion bins: (label, converted)  converted=None means no cut.
CONV_BINS = [
    ('unconverted', False),
    ('converted',   True),
    ('incl',        None),
]

PT_BIN_LABELS   = [label for label, _, _ in PT_BINS]
ETA_BIN_LABELS  = [label for label, _, _ in ETA_BINS]
CONV_BIN_LABELS = [label for label, _ in CONV_BINS]

DEFAULT_REGIONS = [
    ('Preselection', '0L'),
    ('VR',           '0L-mT-mid'),
    ('SR',           '0L-mT-low-loose'),
    ('SR',           '0L-mT-low'),
    ('SR',           '0L-mT-mid-loose'),
    ('SR',           '0L-mT-mid'),
    ('SR',           '0L-mT-hgh-loose'),
    ('SR',           '0L-mT-hgh'),
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
# Accumulation helpers
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


def accumulate(regions):
    """
    Loop over all ntuples and accumulate yields per
    (region, era, lp, process, truth_cat, conv_bin, eta_bin, pt_bin, abcd_bin).

    Returns a nested defaultdict:
      acc[region_key][era][lp][proc][truth][conv][eta][pt][abcd] = {'sw', 'sw2', 'n'}
    """
    def _leaf():
        return {'sw': 0.0, 'sw2': 0.0, 'n': 0, 'sw_pt': 0.0}

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

        pt      = data['ph_pt']
        aeta    = np.abs(data['ph_eta'])
        is_conv = data['ph_conversionType'] > 0

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
                                    w     = totalweight[sel]
                                    sw    = float(w.sum())
                                    sw2   = float((w ** 2).sum())
                                    n     = int(sel.sum())
                                    sw_pt = float((w * pt[sel] / 1e3).sum())  # GeV
                                    for bucket in (
                                        acc[reg_key][era][lp][proc][truth][conv_label][eta_label][pt_label][b],
                                        acc[reg_key][era][lp]['inclusive'][truth][conv_label][eta_label][pt_label][b],
                                    ):
                                        bucket['sw']    += sw
                                        bucket['sw2']   += sw2
                                        bucket['n']     += n
                                        bucket['sw_pt'] += sw_pt

    return acc

# ---------------------------------------------------------------------------
# R' computation and formatting
# ---------------------------------------------------------------------------

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
    return f'{rp:6.3f} \u00b1 {err:5.3f}'


def rp_latex(bins):
    rp, err = rp_val(bins)
    if math.isnan(rp):
        return r'\text{---}'
    return rf'{rp:.3f} \pm {err:.3f}'

# ---------------------------------------------------------------------------
# pT correction helpers
# ---------------------------------------------------------------------------

def avg_pt_val(bins):
    """Weighted average pT (GeV) summed over all ABCD quadrants."""
    sw    = sum(bins[b]['sw']    for b in ('TT', 'TL', 'LT', 'LL'))
    sw_pt = sum(bins[b]['sw_pt'] for b in ('TT', 'TL', 'LT', 'LL'))
    return sw_pt / sw if sw > 0 else float('nan')


def _fit_pts(pts):
    """Fit (x, y) pairs with a line; return (slope, intercept) or None if < 2 pts."""
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return tuple(np.polyfit(xs, ys, 1))


def compute_presel_fits(acc, truth, combine_eras):
    """Linear R'(pT) fits in the preselection region per (era_key, lp).

    Returns dict: (era_key, lp) → {'lo': (slope, intercept) or None,
                                    'hi': (slope, intercept) or None,
                                    'ref_pt': float (GeV)}
    era_key is None when combine_eras=True, otherwise the era string.
    """
    era_groups = [(None, ERAS)] if combine_eras else [(era, [era]) for era in ERAS]
    fits = {}
    for era_key, era_list in era_groups:
        for lp in LP_WPS:
            pts_lo, pts_hi = [], []
            for xl, xc in zip(PT_LABELS, PT_CENTERS):
                bins = merge_bins([
                    acc['Preselection/0L'][era][lp]['inclusive'][truth]['incl']['incl'][xl]
                    for era in era_list
                ])
                rv, _ = rp_val(bins)
                if not math.isnan(rv):
                    (pts_lo if xc <= SPLIT_PT else pts_hi).append((xc, rv))

            ref_bins = merge_bins([
                acc['Preselection/0L'][era][lp]['inclusive'][truth]['incl']['incl']['incl']
                for era in era_list
            ])
            fits[(era_key, lp)] = {
                'lo':     _fit_pts(pts_lo),
                'hi':     _fit_pts(pts_hi),
                'ref_pt': avg_pt_val(ref_bins),
            }
    return fits


def _eval_fit(pt, fit_lo, fit_hi):
    """Evaluate the appropriate linear fit at pt (GeV)."""
    fit = fit_lo if pt <= SPLIT_PT else fit_hi
    if fit is None:
        return float('nan')
    return fit[0] * pt + fit[1]


def get_correction_fn(acc, truth, combine_eras):
    """Return correction_fn(era_label, lp, bins) → float multiplicative factor.

    Corrects R' for the difference in average pT between the given bins and the
    preselection reference:  correction = R'_fit(ref_pt) / R'_fit(avg_pt_bins).
    Returns 1.0 when a correction cannot be computed.
    """
    presel_fits = compute_presel_fits(acc, truth, combine_eras)

    def correction_fn(era_label, lp, bins):
        era_key = None if combine_eras else era_label
        entry   = presel_fits.get((era_key, lp))
        if entry is None:
            return 1.0
        fit_lo, fit_hi, ref_pt = entry['lo'], entry['hi'], entry['ref_pt']
        avg_pt = avg_pt_val(bins)
        if math.isnan(avg_pt) or math.isnan(ref_pt):
            return 1.0
        rp_ref = _eval_fit(ref_pt, fit_lo, fit_hi)
        rp_avg = _eval_fit(avg_pt, fit_lo, fit_hi)
        if math.isnan(rp_ref) or math.isnan(rp_avg) or rp_avg <= 0:
            return 1.0
        return rp_ref / rp_avg

    return correction_fn


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_tables(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins,
                 latex=False):
    if latex:
        _print_tables_latex(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins)
    else:
        _print_tables_text(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins)


def _print_tables_text(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins):
    col_w   = 17
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
                        eta_desc = ('eta: inclusive' if eta_lo is None
                                    else f'|eta| in [{eta_lo}, {eta_hi})')
                        print(f'\n{"="*len(header)}')
                        print(f'{era}  |  {lp}  |  {reg_key}  |  {conv_label}  |  {eta_desc}')
                        print('='*len(header))
                        print(header)
                        for truth in truth_cats:
                            print(sep)
                            for proc in processes:
                                cells = '  '.join(
                                    f'{rp_str(reg_acc[era][lp][proc][truth][conv_label][eta_label][pt_label]):^{col_w}}'
                                    for pt_label, _, _ in PT_BINS
                                )
                                print(f'{truth:<{truth_w}}  {proc:<{proc_w}}  {cells}')
                        print(sep)


def _print_tables_latex(acc, regions, truth_cats, wps, processes, eta_bins, conv_bins):
    col_spec = 'll' + 'c' * len(PT_BINS)

    for rt, rn in regions:
        reg_key = f"{rt}/{rn}"
        reg_acc = acc[reg_key]
        for era in ERAS:
            for lp in wps:
                for conv_label, converted in conv_bins:
                    for eta_label, eta_lo, eta_hi in eta_bins:
                        eta_desc = (r'inclusive $\eta$' if eta_lo is None
                                    else rf'$|\eta| \in [{eta_lo},\,{eta_hi})$')
                        caption = f'{era}, {lp}, {reg_key}, {conv_label}, {eta_desc}'

                        print(r'\begin{table}[htbp]')
                        print(r'\centering')
                        print(rf'\caption{{{caption}}}')
                        print(rf'\begin{{tabular}}{{{col_spec}}}')
                        print(r'\hline')
                        pt_headers = ' & '.join(
                            rf'$p_{{T}}$: {label}' for label, _, _ in PT_BINS)
                        print(rf'Truth & Process & {pt_headers} \\')
                        print(r'\hline')
                        for truth in truth_cats:
                            print(r'\hline')
                            for proc in processes:
                                cells = ' & '.join(
                                    rf'${rp_latex(reg_acc[era][lp][proc][truth][conv_label][eta_label][pt_label])}$'
                                    for pt_label, _, _ in PT_BINS
                                )
                                print(rf'{truth} & {proc} & {cells} \\')
                        print(r'\hline')
                        print(r'\end{tabular}')
                        print(r'\end{table}')
                        print()

# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = 'rprime_analysis_output'
REG_KEY    = 'Preselection/0L'   # used by per-variable plots (pT, eta, conv, process)

# Short x-axis labels for the region plot
REGION_LABELS = {
    'Preselection/0L':    'Presel.',
    'VR/0L-mT-mid':       'VR',
    'SR/0L-mT-low-loose': 'SR-low-l',
    'SR/0L-mT-low':       'SR-low',
    'SR/0L-mT-mid-loose': 'SR-mid-l',
    'SR/0L-mT-mid':       'SR-mid',
    'SR/0L-mT-hgh-loose': 'SR-hgh-l',
    'SR/0L-mT-hgh':       'SR-hgh',
}
REGION_KEYS = [f'{rt}/{rn}' for rt, rn in DEFAULT_REGIONS]

# Color encodes WP (blue=LP4, red=Loose) with lighter shade for Run 2, darker for Run 3.
# Marker encodes era (circle=Run 2, square=Run 3).
SERIES_STYLES = {
    ('LoosePrime4', 'Run2'): {'color': '#7bafd4', 'marker': 'o'},  # light blue
    ('LoosePrime4', 'Run3'): {'color': '#1f77b4', 'marker': 's'},  # dark blue
    ('Loose',       'Run2'): {'color': '#f0908a', 'marker': 'o'},  # light red
    ('Loose',       'Run3'): {'color': '#d62728', 'marker': 's'},  # dark red
}
# Styles for combined Run2+3 series (one per WP)
SERIES_STYLES_COMBINED = {
    'LoosePrime4': {'color': '#1f77b4', 'marker': 'D'},  # blue diamond
    'Loose':       {'color': '#d62728', 'marker': 'D'},  # red diamond
}

# pT boundary (GeV) separating the two linear-fit regions
SPLIT_PT = 25.

# Colors for per-WP fit lines (WP only, not era-specific)
FIT_WP_COLORS = {
    'LoosePrime4': '#1f77b4',  # dark blue
    'Loose':       '#d62728',  # dark red
}

# Exclusive-only labels (strip 'incl') for each dimension
PT_LABELS   = [l for l, _, _ in PT_BINS   if l != 'incl']
ETA_LABELS  = [l for l, _, _ in ETA_BINS  if l != 'incl']
CONV_LABELS = [l for l, _    in CONV_BINS if l != 'incl']
PROC_LABELS = ['Znunu', 'Wtaunu', 'inclusive']

# Bin centres for the pT plot (GeV).  For the open-ended last bin (>80)
# use 100 GeV as a representative display position.
PT_CENTERS = [
    0.5 * (lo + hi) if hi is not None else lo + 20.
    for label, lo, hi in PT_BINS if label != 'incl'
]

# Reverse map: short label → full region key, for the region plot
_LABEL_TO_KEY = {REGION_LABELS[k]: k for k in REGION_KEYS}


def build_variables(truth):
    """Return the VARIABLES list for the given truth category label."""
    presel_sub = f'Preselection / 0L  —  {truth}'
    return [
        (
            'pT',
            PT_LABELS,
            r'$p_{\rm T}^{\gamma}$ [GeV]',
            lambda acc, era, lp, xl, t=truth:
                acc[REG_KEY][era][lp]['inclusive'][t]['incl']['incl'][xl],
            PT_CENTERS,
            presel_sub,
        ),
        (
            'eta',
            ETA_LABELS,
            r'$|\eta^{\gamma}|$ region',
            lambda acc, era, lp, xl, t=truth:
                acc[REG_KEY][era][lp]['inclusive'][t]['incl'][xl]['incl'],
            None,
            presel_sub,
        ),
        (
            'conversion',
            CONV_LABELS,
            'Conversion status',
            lambda acc, era, lp, xl, t=truth:
                acc[REG_KEY][era][lp]['inclusive'][t][xl]['incl']['incl'],
            None,
            presel_sub,
        ),
        (
            'process',
            PROC_LABELS,
            'Process',
            lambda acc, era, lp, xl, t=truth:
                acc[REG_KEY][era][lp][xl][t]['incl']['incl']['incl'],
            None,
            presel_sub,
        ),
        (
            'region',
            [REGION_LABELS[k] for k in REGION_KEYS],
            'Region',
            lambda acc, era, lp, xl, t=truth:
                acc[_LABEL_TO_KEY[xl]][era][lp]['inclusive'][t]['incl']['incl']['incl'],
            None,
            f'All regions  —  {truth}  —  inclusive',
        ),
    ]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_bins(bins_list):
    """Sum sw, sw2, and sw_pt across a list of bins dicts (each with TT/TL/LT/LL keys)."""
    merged = {b: {'sw': 0., 'sw2': 0., 'sw_pt': 0.} for b in ('TT', 'TL', 'LT', 'LL')}
    for bins in bins_list:
        for b in ('TT', 'TL', 'LT', 'LL'):
            merged[b]['sw']    += bins[b]['sw']
            merged[b]['sw2']   += bins[b]['sw2']
            merged[b]['sw_pt'] += bins[b].get('sw_pt', 0.)
    return merged


# ---------------------------------------------------------------------------
# pT linear fit
# ---------------------------------------------------------------------------

def add_pt_fits(ax, acc, x_labels, pt_centers, get_bins_fn, combine_eras, y_top=0.82):
    """Fit R' vs pT linearly in [lo, SPLIT_PT] and [SPLIT_PT, hi] per WP.

    Points from all eras are pooled together for each WP fit.
    Draws dotted fit lines on ax.  Labels are placed in a 2×2 grid just below
    the ATLAS/subtitle block:  rows = WP (Loose top, LP4 bottom),
                               cols = pT range (low left, high right).
    """
    # grid[(row, col)] = (color, text)
    # row 0 = Loose (top), row 1 = LoosePrime4 (bottom)
    # col 0 = low pT (left), col 1 = high pT (right)
    WP_ROW = {'Loose': 0, 'LoosePrime4': 1}
    grid = {}

    for lp in LP_WPS:
        color = FIT_WP_COLORS[lp]
        pts_lo, pts_hi = [], []

        for xl, xc in zip(x_labels, pt_centers):
            if combine_eras:
                candidates = [merge_bins([get_bins_fn(acc, era, lp, xl) for era in ERAS])]
            else:
                candidates = [get_bins_fn(acc, era, lp, xl) for era in ERAS]
            for bins in candidates:
                rv, _ = rp_val(bins)
                if not math.isnan(rv):
                    (pts_lo if xc <= SPLIT_PT else pts_hi).append((xc, rv))

        for col, (pts, x_draw, range_str) in enumerate([
            (pts_lo, (min(pt_centers), SPLIT_PT),       f'pT ≤ {SPLIT_PT:.0f} GeV'),
            (pts_hi, (SPLIT_PT,        max(pt_centers)), f'pT > {SPLIT_PT:.0f} GeV'),
        ]):
            if len(pts) < 2:
                continue
            xs_f = np.array([p[0] for p in pts])
            ys_f = np.array([p[1] for p in pts])
            slope, intercept = np.polyfit(xs_f, ys_f, 1)

            x_line = np.linspace(x_draw[0], x_draw[1], 200)
            ax.plot(x_line, slope * x_line + intercept,
                    color=color, linestyle=':', linewidth=1.8, alpha=0.9, zorder=4)

            i_sign = '+' if intercept >= 0 else '-'
            s_sign = '+' if slope     >= 0 else '-'
            eq = (f'{lp}  [{range_str}]\n'
                  f"  R' = {i_sign}{abs(intercept):.3f} {s_sign} {abs(slope):.4f}·pT")
            grid[(WP_ROW[lp], col)] = (color, eq)

    # Render 2×2 grid just below the ATLAS/subtitle block
    x_cols = [0.05, 0.53]
    y_rows = [y_top, y_top - 0.08]

    for (row, col), (color, text) in grid.items():
        ax.text(x_cols[col], y_rows[row], text,
                transform=ax.transAxes, fontsize=7, va='top',
                color=color, fontfamily='monospace')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(acc, var_tag, x_labels, xlabel, get_bins_fn, x_coords=None, subtitle='',
              truth='JFP', y_max=None, combine_eras=False, fit_pt=False,
              correction_fn=None, suffix=''):
    """Single-panel figure with four (era × WP) series, or two WP series if combine_eras.

    x_coords: if provided, use as numeric x positions (continuous axis);
              otherwise use equally-spaced categorical positions.
    y_max: if provided, clip y-axis at this value and draw upward arrows for
           any off-scale points.
    combine_eras: if True, sum Run2+Run3 yields before computing R'.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    continuous = x_coords is not None
    xs         = np.array(x_coords, dtype=float) if continuous else np.arange(len(x_labels), dtype=float)
    n_series   = len(LP_WPS) if combine_eras else len(ERAS) * len(LP_WPS)
    # For categorical axes jitter points so errorbars don't overlap; no jitter for continuous
    jitter = np.zeros(n_series) if continuous else np.linspace(-0.15, 0.15, n_series)

    data_lo  = []   # rv - re for plotted points (for range calculation)
    data_hi  = []   # rv + re for plotted points
    off_scale = []  # (x_pos, color) for points beyond y_max

    idx = 0
    if combine_eras:
        for lp in reversed(LP_WPS):
            xpos, yval, yerr = [], [], []
            sty = SERIES_STYLES_COMBINED[lp]
            for j, xl in enumerate(x_labels):
                bins = merge_bins([get_bins_fn(acc, era, lp, xl) for era in ERAS])
                rv, re = _rp_corrected(bins, correction_fn, None, lp)
                if not math.isnan(rv):
                    if y_max is not None and rv > y_max:
                        off_scale.append((xs[j] + jitter[idx], sty['color']))
                    else:
                        xpos.append(xs[j] + jitter[idx])
                        yval.append(rv)
                        yerr.append(re)
                        data_lo.append(rv - re)
                        data_hi.append(rv + re)
            if yval:
                ax.errorbar(xpos, yval, yerr=yerr,
                            fmt=sty['marker'], color=sty['color'],
                            label=lp,
                            capsize=3, markersize=5, linewidth=1)
            idx += 1
    else:
        for lp in reversed(LP_WPS):
            for era in ERAS:
                xpos, yval, yerr = [], [], []
                sty = SERIES_STYLES[(lp, era)]
                for j, xl in enumerate(x_labels):
                    bins = get_bins_fn(acc, era, lp, xl)
                    rv, re = _rp_corrected(bins, correction_fn, era, lp)
                    if not math.isnan(rv):
                        if y_max is not None and rv > y_max:
                            off_scale.append((xs[j] + jitter[idx], sty['color']))
                        else:
                            xpos.append(xs[j] + jitter[idx])
                            yval.append(rv)
                            yerr.append(re)
                            data_lo.append(rv - re)
                            data_hi.append(rv + re)
                if yval:
                    ax.errorbar(xpos, yval, yerr=yerr,
                                fmt=sty['marker'], color=sty['color'],
                                label=f'{lp} / {era}',
                                capsize=3, markersize=5, linewidth=1)
                idx += 1

    # pT linear fits (only for the pT plot, when requested)
    if fit_pt and var_tag == 'pT' and x_coords is not None:
        # shift fit labels down an extra row when combine_eras adds a line to the header block
        add_pt_fits(ax, acc, x_labels, list(xs), get_bins_fn, combine_eras,
                    y_top=0.82 - (0.07 if combine_eras else 0.))

    # Y-axis: default range [0.9, 1.8]; expand if data falls outside; cap at y_max
    DEFAULT_YMIN, DEFAULT_YMAX = 0.9, 1.8
    if data_lo:
        lo = min(data_lo)
        hi = max(data_hi)
        ymin_new = min(DEFAULT_YMIN, max(0., lo))
        ymax_raw = max(DEFAULT_YMAX, hi + 0.25 * (hi - ymin_new))
    else:
        ymin_new, ymax_raw = DEFAULT_YMIN, DEFAULT_YMAX
    ymax_new = y_max if (y_max is not None and off_scale) else ymax_raw
    ax.set_ylim(ymin_new, ymax_new)

    # Draw upward arrows for off-scale points
    if off_scale:
        final_ymin, final_ymax = ax.get_ylim()
        arrow_head = final_ymin + 0.94 * (final_ymax - final_ymin)
        arrow_tail = final_ymin + 0.80 * (final_ymax - final_ymin)
        for x_off, color in off_scale:
            ax.annotate('',
                        xy=(x_off, arrow_head),
                        xytext=(x_off, arrow_tail),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.5, mutation_scale=12))

    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', color='lightgray', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    if not continuous:
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=13, loc='right')
    ax.set_ylabel("R'", fontsize=13, loc='top')
    ax.tick_params(axis='both', which='both', direction='in')
    if continuous:
        ax.set_xlim(10., xs[-1] + 0.05 * (xs[-1] - xs[0]))
        ticks = ax.get_xticks()
        if 10. not in ticks:
            ax.set_xticks(np.sort(np.append(ticks[(ticks >= 10.) & (ticks <= xs[-1])], 10.)))
    else:
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)

    # ATLAS label
    t_atlas = ax.text(0.05, 0.97, 'ATLAS',
                      transform=ax.transAxes, fontsize=13, va='top',
                      fontweight='bold', fontstyle='italic')
    fig.canvas.draw()
    bb      = t_atlas.get_window_extent(renderer=fig.canvas.get_renderer())
    x1_axes = ax.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1_axes + 0.01, 0.97, 'Internal',
            transform=ax.transAxes, fontsize=13, va='top')
    y_next = 0.90
    if combine_eras:
        ax.text(0.05, y_next, 'Run 2 + Run 3',
                transform=ax.transAxes, fontsize=10, va='top')
        y_next -= 0.07
    ax.text(0.05, y_next, subtitle,
            transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace')
    if suffix:
        y_next -= 0.07
        ax.text(0.05, y_next, f'({suffix})',
                transform=ax.transAxes, fontsize=9, va='top', fontstyle='italic')

    ax.legend(fontsize=9, framealpha=0.85, loc='upper right',
              bbox_to_anchor=(0.97, 0.97), ncol=1 if combine_eras else 2)

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    era_tag  = 'Run2p3' if combine_eras else 'byera'
    file_sfx = f'_{suffix}' if suffix else ''
    out = os.path.join(OUTPUT_DIR, f'rprime_vs_{var_tag}_{truth}_{era_tag}{file_sfx}')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}.{{pdf,png}}')


# ---------------------------------------------------------------------------
# Per-variable table output (terminal and LaTeX)
# ---------------------------------------------------------------------------

def _per_var_series(combine_eras):
    """Return (lp, era_label) pairs and column header strings for one table."""
    if combine_eras:
        return ([(lp, 'Run2+3') for lp in LP_WPS],
                [f'{lp} / Run2+3' for lp in LP_WPS])
    else:
        return ([(lp, era) for lp in LP_WPS for era in ERAS],
                [f'{lp} / {era}' for lp in LP_WPS for era in ERAS])


def _get_bins(acc, lp, era_label, xl, get_bins_fn, combine_eras):
    if combine_eras:
        return merge_bins([get_bins_fn(acc, era, lp, xl) for era in ERAS])
    return get_bins_fn(acc, era_label, lp, xl)


def _rp_corrected(bins, correction_fn, era_label, lp):
    """Return (rp, err) with optional multiplicative pT correction applied."""
    rp, err = rp_val(bins)
    if correction_fn is None or math.isnan(rp):
        return rp, err
    c = correction_fn(era_label, lp, bins)
    return rp * c, err * c


def print_table_text(acc, var_tag, x_labels, xlabel, get_bins_fn,
                     truth='JFP', subtitle='', combine_eras=False,
                     correction_fn=None, suffix=''):
    """Print a terminal-friendly R' table for one variable."""
    series, col_headers = _per_var_series(combine_eras)
    col_w  = 17
    row_w  = max(len(xl) for xl in x_labels) + 2

    title = subtitle + (f'  [{suffix}]' if suffix else '')
    header = f'{"":>{row_w}}  ' + '  '.join(f'{h:^{col_w}}' for h in col_headers)
    sep    = '-' * len(header)

    print(f'\n{title}')
    print(sep)
    print(header)
    print(sep)
    for xl in x_labels:
        cells = []
        for lp, era_label in series:
            bins = _get_bins(acc, lp, era_label, xl, get_bins_fn, combine_eras)
            rp, err = _rp_corrected(bins, correction_fn, era_label, lp)
            if math.isnan(rp):
                cells.append(f'{"      ---      ":^{col_w}}')
            else:
                cells.append(f'{f"{rp:6.3f} \u00b1 {err:5.3f}":^{col_w}}')
        print(f'{xl:>{row_w}}  {"  ".join(cells)}')
    print(sep)


def write_table_tex(acc, var_tag, x_labels, xlabel, get_bins_fn,
                    truth='JFP', subtitle='', combine_eras=False,
                    correction_fn=None, suffix=''):
    """Write a .tex file with an R' table for one variable, mirroring the plot."""
    series, col_headers = _per_var_series(combine_eras)
    col_spec = 'l' + 'c' * len(col_headers)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    era_tag  = 'Run2p3' if combine_eras else 'byera'
    file_sfx = f'_{suffix}' if suffix else ''
    out = os.path.join(OUTPUT_DIR, f'rprime_vs_{var_tag}_{truth}_{era_tag}{file_sfx}.tex')

    cap_base = xlabel.replace('$','').replace('{','').replace('}','')
    cap_sfx  = f' ({suffix})' if suffix else ''

    with open(out, 'w') as f:
        def w(s=''):
            f.write(s + '\n')

        w(r'\begin{table}[htbp]')
        w(r'\centering')
        w(rf"\caption{{R' vs {cap_base}{cap_sfx} — {subtitle}}}")
        w(rf'\begin{{tabular}}{{{col_spec}}}')
        w(r'\hline')
        w(f"{xlabel} & {' & '.join(col_headers)} \\\\")
        w(r'\hline')
        for xl in x_labels:
            cells = []
            for lp, era_label in series:
                bins = _get_bins(acc, lp, era_label, xl, get_bins_fn, combine_eras)
                rp, err = _rp_corrected(bins, correction_fn, era_label, lp)
                if math.isnan(rp):
                    cells.append(r'$\text{---}$')
                else:
                    cells.append(rf'${rp:.3f} \pm {err:.3f}$')
            w(f"{xl} & {' & '.join(cells)} \\\\")
        w(r'\hline')
        w(r'\end{tabular}')
        w(r'\end{table}')

    print(f'Saved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="R' summary plots and/or tables vs pT, eta, conversion, process, region.")

    # ---- plot options --------------------------------------------------------
    plot_grp = parser.add_argument_group('plot options')
    plot_grp.add_argument(
        '--truth', metavar='CAT', default='JFP',
        choices=TRUTH_CATS,
        help=f'Truth category for plots (default: JFP). Choices: {", ".join(TRUTH_CATS)}')
    plot_grp.add_argument(
        '--y-max', type=float, default=None, metavar='VAL',
        help='Cap plot y-axis at this value; off-scale points shown as upward arrows.')
    plot_grp.add_argument(
        '--combine-eras', action='store_true',
        help="Combine Run2+Run3 yields into a single inclusive R' per WP.")
    plot_grp.add_argument(
        '--no-plots', action='store_true',
        help='Skip plot generation (useful with --fulltables or --latex alone).')
    plot_grp.add_argument(
        '--fit-pt', action='store_true',
        help=(f'Fit R\' vs pT with a linear model in two ranges '
              f'(pT ≤ {SPLIT_PT:.0f} GeV and pT > {SPLIT_PT:.0f} GeV), '
              f'separately per WP. Draws fit lines and equations on the pT plot.'))
    plot_grp.add_argument(
        '--correct-pt', action='store_true',
        help=('Also produce pT-corrected plots and tables.  R\' values are rescaled by '
              'R\'_fit(<pT>_presel) / R\'_fit(<pT>_bin) to remove the pT dependence '
              'seen across regions.  Corrected outputs get a "_ptcorr" filename suffix.'))

    # ---- table options -------------------------------------------------------
    tbl_grp = parser.add_argument_group('table options')
    tbl_grp.add_argument(
        '--tables', action='store_true',
        help="Print per-variable R' summary tables to terminal (mirrors the plots).")
    tbl_grp.add_argument(
        '--fulltables', action='store_true',
        help="Print full R' breakdown tables to terminal (truth, process, pT, WP, era, region).")
    tbl_grp.add_argument(
        '--latex', action='store_true',
        help='Write per-variable R\' tables as .tex files in the output directory.')
    tbl_grp.add_argument(
        '--table-truth', nargs='+', metavar='CAT', choices=TRUTH_CATS, default=None,
        help=f'Truth categories for tables (default: all). Choices: {", ".join(TRUTH_CATS)}')
    tbl_grp.add_argument(
        '--wp', nargs='+', metavar='WP', choices=LP_WPS, default=None,
        help=f'ID working points for tables (default: all). Choices: {", ".join(LP_WPS)}')
    tbl_grp.add_argument(
        '--processes', nargs='+', metavar='PROC', choices=ALL_PROCESSES, default=None,
        help=f'Processes for tables (default: all). Choices: {", ".join(ALL_PROCESSES)}')
    tbl_grp.add_argument(
        '--eta', nargs='+', metavar='ETA', choices=ETA_BIN_LABELS, default=None,
        help=f'Eta bins for tables (default: all). Choices: {", ".join(ETA_BIN_LABELS)}')
    tbl_grp.add_argument(
        '--conv', nargs='+', metavar='CONV', choices=CONV_BIN_LABELS, default=None,
        help=f'Conversion categories for tables (default: all). Choices: {", ".join(CONV_BIN_LABELS)}')
    tbl_grp.add_argument(
        '--regions', nargs='+', metavar='TYPE/NAME', default=None,
        help='Regions for tables (default: all). Format: TYPE/NAME e.g. Preselection/0L')

    args = parser.parse_args()

    # ---- resolve table filter arguments -------------------------------------
    table_regions = DEFAULT_REGIONS
    if args.regions:
        table_regions = []
        for r in args.regions:
            parts = r.split('/', 1)
            if len(parts) != 2:
                parser.error(f'Region must be TYPE/NAME, got: {r!r}')
            table_regions.append((parts[0], parts[1]))

    table_truth    = args.table_truth or TRUTH_CATS
    table_wps      = args.wp          or LP_WPS
    table_processes= args.processes   or ALL_PROCESSES
    table_eta      = [(l, lo, hi) for l, lo, hi in ETA_BINS
                      if l in (args.eta  or ETA_BIN_LABELS)]
    table_conv     = [(l, c)      for l, c      in CONV_BINS
                      if l in (args.conv or CONV_BIN_LABELS)]

    # ---- accumulate (always use all regions so plots are complete) ----------
    print(f'Accumulating ntuples for all regions ...')
    acc = accumulate(DEFAULT_REGIONS)

    # pre-compute pT correction function once (only if needed)
    corr_fn = get_correction_fn(acc, args.truth, args.combine_eras) if args.correct_pt else None

    # ---- plots, terminal summary tables, and/or latex table files ------------
    if not args.no_plots or args.tables or args.latex:
        for var_tag, x_labels, xlabel, get_bins_fn, x_coords, subtitle in build_variables(args.truth):
            # uncorrected
            if not args.no_plots:
                make_plot(acc, var_tag, x_labels, xlabel, get_bins_fn, x_coords, subtitle,
                          truth=args.truth, y_max=args.y_max, combine_eras=args.combine_eras,
                          fit_pt=args.fit_pt)
            if args.tables:
                print_table_text(acc, var_tag, x_labels, xlabel, get_bins_fn,
                                 truth=args.truth, subtitle=subtitle,
                                 combine_eras=args.combine_eras)
            if args.latex:
                write_table_tex(acc, var_tag, x_labels, xlabel, get_bins_fn,
                                truth=args.truth, subtitle=subtitle,
                                combine_eras=args.combine_eras)
            # pT-corrected (only when requested)
            if args.correct_pt:
                if not args.no_plots:
                    make_plot(acc, var_tag, x_labels, xlabel, get_bins_fn, x_coords, subtitle,
                              truth=args.truth, y_max=args.y_max, combine_eras=args.combine_eras,
                              fit_pt=args.fit_pt, correction_fn=corr_fn, suffix='ptcorr')
                if args.tables:
                    print_table_text(acc, var_tag, x_labels, xlabel, get_bins_fn,
                                     truth=args.truth, subtitle=subtitle,
                                     combine_eras=args.combine_eras,
                                     correction_fn=corr_fn, suffix='ptcorr')
                if args.latex:
                    write_table_tex(acc, var_tag, x_labels, xlabel, get_bins_fn,
                                    truth=args.truth, subtitle=subtitle,
                                    combine_eras=args.combine_eras,
                                    correction_fn=corr_fn, suffix='ptcorr')

    # ---- full terminal breakdown tables --------------------------------------
    if args.fulltables:
        print_tables(acc, table_regions, table_truth, table_wps, table_processes,
                     table_eta, table_conv, latex=False)
