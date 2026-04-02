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
matplotlib.rcParams['font.family'] = 'Nimbus Sans'
import matplotlib.pyplot as plt
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
    """Sum sw and sw2 across a list of bins dicts (each with TT/TL/LT/LL keys)."""
    merged = {b: {'sw': 0., 'sw2': 0.} for b in ('TT', 'TL', 'LT', 'LL')}
    for bins in bins_list:
        for b in ('TT', 'TL', 'LT', 'LL'):
            merged[b]['sw']  += bins[b]['sw']
            merged[b]['sw2'] += bins[b]['sw2']
    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(acc, var_tag, x_labels, xlabel, get_bins_fn, x_coords=None, subtitle='',
              truth='JFP', y_max=None, combine_eras=False):
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
        for lp in LP_WPS:
            xpos, yval, yerr = [], [], []
            sty = SERIES_STYLES_COMBINED[lp]
            for j, xl in enumerate(x_labels):
                bins = merge_bins([get_bins_fn(acc, era, lp, xl) for era in ERAS])
                rv, re = rp_val(bins)
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
                            label=f'{lp} / Run2+3',
                            capsize=3, markersize=5, linewidth=1)
            idx += 1
    else:
        for lp in LP_WPS:
            for era in ERAS:
                xpos, yval, yerr = [], [], []
                sty = SERIES_STYLES[(lp, era)]
                for j, xl in enumerate(x_labels):
                    bins = get_bins_fn(acc, era, lp, xl)
                    rv, re = rp_val(bins)
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
    ax.text(0.05, 0.90, subtitle,
            transform=ax.transAxes, fontsize=10, va='top', fontfamily='monospace')

    ax.legend(fontsize=9, framealpha=0.85, loc='upper right',
              bbox_to_anchor=(0.97, 0.97), ncol=2)

    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    era_tag = 'Run2p3' if combine_eras else 'byera'
    out = os.path.join(OUTPUT_DIR, f'rprime_vs_{var_tag}_{truth}_{era_tag}')
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


def print_table_text(acc, var_tag, x_labels, xlabel, get_bins_fn,
                     truth='JFP', subtitle='', combine_eras=False):
    """Print a terminal-friendly R' table for one variable."""
    series, col_headers = _per_var_series(combine_eras)
    col_w  = 17
    row_w  = max(len(xl) for xl in x_labels) + 2

    header = f'{"":>{row_w}}  ' + '  '.join(f'{h:^{col_w}}' for h in col_headers)
    sep    = '-' * len(header)

    print(f'\n{subtitle}')
    print(sep)
    print(header)
    print(sep)
    for xl in x_labels:
        cells = '  '.join(
            f'{rp_str(_get_bins(acc, lp, era_label, xl, get_bins_fn, combine_eras)):^{col_w}}'
            for lp, era_label in series
        )
        print(f'{xl:>{row_w}}  {cells}')
    print(sep)

def write_table_tex(acc, var_tag, x_labels, xlabel, get_bins_fn,
                    truth='JFP', subtitle='', combine_eras=False):
    """Write a .tex file with an R' table for one variable, mirroring the plot."""
    series, col_headers = _per_var_series(combine_eras)
    col_spec = 'l' + 'c' * len(col_headers)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    era_tag = 'Run2p3' if combine_eras else 'byera'
    out = os.path.join(OUTPUT_DIR, f'rprime_vs_{var_tag}_{truth}_{era_tag}.tex')

    with open(out, 'w') as f:
        def w(s=''):
            f.write(s + '\n')

        w(r'\begin{table}[htbp]')
        w(r'\centering')
        w(rf"\caption{{R' vs {xlabel.replace('$','').replace('{','').replace('}','')} — {subtitle}}}")
        w(rf'\begin{{tabular}}{{{col_spec}}}')
        w(r'\hline')
        w(f"{xlabel} & {' & '.join(col_headers)} \\\\")
        w(r'\hline')
        for xl in x_labels:
            cells = [f'${rp_latex(_get_bins(acc, lp, era_label, xl, get_bins_fn, combine_eras))}$'
                     for lp, era_label in series]
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

    # ---- plots, terminal summary tables, and/or latex table files ------------
    if not args.no_plots or args.tables or args.latex:
        for var_tag, x_labels, xlabel, get_bins_fn, x_coords, subtitle in build_variables(args.truth):
            if not args.no_plots:
                make_plot(acc, var_tag, x_labels, xlabel, get_bins_fn, x_coords, subtitle,
                          truth=args.truth, y_max=args.y_max, combine_eras=args.combine_eras)
            if args.tables:
                print_table_text(acc, var_tag, x_labels, xlabel, get_bins_fn,
                                 truth=args.truth, subtitle=subtitle,
                                 combine_eras=args.combine_eras)
            if args.latex:
                write_table_tex(acc, var_tag, x_labels, xlabel, get_bins_fn,
                                truth=args.truth, subtitle=subtitle,
                                combine_eras=args.combine_eras)

    # ---- full terminal breakdown tables --------------------------------------
    if args.fulltables:
        print_tables(acc, table_regions, table_truth, table_wps, table_processes,
                     table_eta, table_conv, latex=False)
