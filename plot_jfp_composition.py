#!/usr/bin/env python3
"""
Diagnostic stacked-histogram plots of JFP and Other photon backgrounds,
split by process (Znunu, Wtaunu, Wmunu, Wenu) and truth category (JFP, Other).

Produces two types of output:
  1. Per-variable stacked distributions for each (region, ID, era).
  2. Per-region process-composition summaries (absolute yield + fraction) for
     each (ID, truth cat, era), showing which processes populate each region.

Regions: Preselection/0L, VR/0L-mT-mid, all SR-0L-mT-{low,mid,hgh} variants.
Run 2: mc20 files.  Run 3: mc23 files.

Output: jfp_composition_plots/<tag>.{pdf,png}
"""

import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'FreeSans', 'DejaVu Sans']
import matplotlib.ticker

import matplotlib.pyplot as plt
import numpy as np
import uproot
import os
import glob

from abcd_utils import get_region_masks, LoosePrimeDefs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = "/data/mhance/SUSY/ntuples/v4.2"
OUTPUT_DIR = "jfp_composition_plots"

PROCESSES  = ['Znunu', 'Wtaunu', 'Wmunu', 'Wenu']
_OTHER_MC  = 'OtherMC'
TRUTH_CATS = ['JFP', 'Other']

# Files to exclude even when --include-other-mc is set
_SKIP = ('gammajet', 'jetjet', '_jj_', 'N2_', 'signal', 'data_')

# ABCD quadrant categories (Tight ID vs LP4, tight iso vs loose iso)
ID_CATS = {
    'TT': 'Tight ID, tight iso',
    'TL': 'Tight ID, loose iso',
    'LT': 'LP4 (not Tight), tight iso',
    'LL': 'LP4 (not Tight), loose iso',
}

# Stacking order: all JFP (bottom group) then all Other (top group)
# Within each group: Wenu (bottom), Wmunu, Wtaunu, Znunu (top)
_STACK_PROCS = ['Wenu', 'Wmunu', 'Wtaunu', 'Znunu']
STACK_ORDER  = [(proc, tc) for tc in TRUTH_CATS for proc in _STACK_PROCS]

# Per-process colors: darker shade for JFP, lighter for Other
PROCESS_COLORS = {
    'Znunu':   {'JFP': '#1f77b4', 'Other': '#aec7e8'},   # blue
    'Wtaunu':  {'JFP': '#d62728', 'Other': '#f5a9a9'},   # red
    'Wmunu':   {'JFP': '#2ca02c', 'Other': '#aee4ae'},   # green
    'Wenu':    {'JFP': '#9467bd', 'Other': '#c5b0d5'},   # purple
    'OtherMC': {'JFP': '#7f7f7f', 'Other': '#c7c7c7'},   # gray
}

REGIONS = [
    ('Preselection', '0L'),
    ('VR',           '0L-mT-mid'),
    ('SR',           '0L-mT-low-loose'),
    ('SR',           '0L-mT-low'),
    ('SR',           '0L-mT-mid-loose'),
    ('SR',           '0L-mT-mid'),
    ('SR',           '0L-mT-hgh-loose'),
    ('SR',           '0L-mT-hgh'),
]

REGION_LABELS = {
    'Preselection/0L':    'Preselection 0L',
    'VR/0L-mT-mid':       'VR 0L-mT-mid',
    'SR/0L-mT-low-loose': 'SR 0L-mT-low-loose',
    'SR/0L-mT-low':       'SR 0L-mT-low',
    'SR/0L-mT-mid-loose': 'SR 0L-mT-mid-loose',
    'SR/0L-mT-mid':       'SR 0L-mT-mid',
    'SR/0L-mT-hgh-loose': 'SR 0L-mT-hgh-loose',
    'SR/0L-mT-hgh':       'SR 0L-mT-hgh',
}

# Truth categories for the composition summary plots (JFP+Other = sum of both)
TRUTH_COMP_CATS = ['JFP', 'Other', 'JFP+Other']

# Variable used to compute total yields for composition plots
_YIELD_VAR = 'ph_pt'

# Variable definitions:
#   bins  — histogram bin edges
#   label — x-axis label (LaTeX OK)
#   norm  — branch name to divide by (for iso/pT ratios), or None
#   scale — multiplicative scale applied to the raw branch value before histogramming
#           (and before the norm division if norm is set)
VARIABLES = {
    'ph_pt': {
        'bins':  np.arange(0., 100., 5.),
        'label': r'$p_{\rm T}^{\gamma}$ [GeV]',
        'norm':  None,
        'scale': 1e-3,   # MeV → GeV
    },
    'ph_eta': {
        'bins':  np.arange(-3., 3.05, 0.2),
        'label': r'$\eta^{\gamma}$',
        'norm':  None,
        'scale': 1.,
    },
    'ph_topoetcone40': {
        'bins':  np.arange(-0.2, 5.01, 0.1),
        'label': r'$\mathrm{topo}E_{\rm T}^{\rm cone40}\ /\ p_{\rm T}^{\gamma}$',
        'norm':  'ph_pt',   # both branches in MeV → ratio is dimensionless
        'scale': 1.,
    },
    'ph_topoetcone20': {
        'bins':  np.arange(-0.2, 5.01, 0.1),
        'label': r'$\mathrm{topo}E_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
        'norm':  'ph_pt',
        'scale': 1.,
    },
    'ph_ptcone20': {
        'bins':  np.arange(0., 5.01, 0.1),
        'label': r'$p_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
        'norm':  'ph_pt',
        'scale': 1.,
    },
    'ph_truthOrigin': {
        'bins':  np.arange(-0.5, 50.5, 1.),
        'label': r'photon truthOrigin',
        'norm':  None,
        'scale': 1.,
    },
    'ph_truthType': {
        'bins':  np.arange(-0.5, 25.5, 1.),
        'label': r'photon truthType',
        'norm':  None,
        'scale': 1.,
    },
    'dPhiGammaMet': {
        'bins':  np.arange(0., 3.25, 0.15),
        'label': r'$\Delta\phi(\gamma, E_{\rm T}^{\rm miss})$',
        'norm':  None,
        'scale': 1.,
    },
    'dPhiGammaJ1': {
        'bins':  np.arange(0., 3.25, 0.15),
        'label': r'$\Delta\phi(\gamma, j_1)$',
        'norm':  None,
        'scale': 1.,
    },
    'mTGammaMet': {
        'bins':  np.arange(0., 205., 10.),
        'label': r'$m_{\rm T}(\gamma, E_{\rm T}^{\rm miss})$ [GeV]',
        'norm':  None,
        'scale': 1e-3,   # MeV → GeV
    },
}

NEEDED_BRANCHES = [
    'ph_pt', 'ph_eta', 'ph_ptcone20', 'ph_topoetcone20', 'ph_topoetcone40',
    'ph_truthOrigin', 'ph_truthType', 'ph_truthJFP', 'ph_truthother',
    'ph_select_baseline', 'ph_select_tightID', 'ph_select_hybridCOIso', 'ph_isEM',
    'weight_total', 'weight_fjvt_effSF', 'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
    'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nPhotons_baseline', 'nPhotons_skims', 'nPhotons_baseline_noOR',
    'mTGammaMet', 'dPhiGammaMet', 'dPhiGammaJ1', 'nTau20_veryloose', 'met_signif',
]

LP4_MASK_BIT = LoosePrimeDefs['LoosePrime4']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def classify_file(fname, include_other_mc=False):
    """Return (process, is_run2) for a filename, or None if not a target file."""
    if 'mc20' in fname:
        run2 = True
    elif 'mc23' in fname:
        run2 = False
    else:
        return None
    if any(s in fname for s in _SKIP):
        return None
    for proc in PROCESSES:
        if proc in fname:
            return proc, run2
    if include_other_mc:
        return _OTHER_MC, run2
    return None


def get_files(run2, include_other_mc=False):
    """Return list of (filepath, process) for the requested run era."""
    result = []
    for fp in sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root'))):
        info = classify_file(os.path.basename(fp), include_other_mc)
        if info is None:
            continue
        proc, frun2 = info
        if frun2 != run2:
            continue
        result.append((fp, proc))
    return result

# ---------------------------------------------------------------------------
# Histogram accumulation
# ---------------------------------------------------------------------------

def _zero_hists(procs):
    """Initialise zero-filled histogram arrays for all regions/id_cats/processes/truth_cats/vars."""
    return {
        f"{rt}/{rn}": {
            id_cat: {
                proc: {
                    tc: {var: np.zeros(len(cfg['bins']) - 1) for var, cfg in VARIABLES.items()}
                    for tc in TRUTH_CATS
                }
                for proc in procs
            }
            for id_cat in ID_CATS
        }
        for rt, rn in REGIONS
    }


def accumulate(run2, include_other_mc=False):
    """Loop over all relevant MC files and fill histograms."""
    procs = PROCESSES + ([_OTHER_MC] if include_other_mc else [])
    hists = _zero_hists(procs)
    sumw2 = _zero_hists(procs)

    files = get_files(run2, include_other_mc)
    era   = 'Run 2' if run2 else 'Run 3'
    print(f"\nAccumulating {era}: {len(files)} files")

    for fp, proc in files:
        print(f"  {proc:<8}  {os.path.basename(fp)}")
        try:
            with uproot.open(fp) as uf:
                if 'picontuple' not in uf:
                    print("    (no picontuple tree, skipping)")
                    continue
                data = uf['picontuple'].arrays(NEEDED_BRANCHES, library='np')
        except Exception as exc:
            print(f"    ERROR reading file: {exc}")
            continue

        totalweight = (
            data['weight_total'] *
            data['weight_fjvt_effSF'] *
            data['weight_ftag_effSF_GN2v01_Continuous'] *
            data['weight_jvt_effSF']
        )

        baseline  = data['ph_select_baseline']     == 1
        is_tight  = data['ph_select_tightID']      == 1
        is_iso    = data['ph_select_hybridCOIso']  == 1
        is_lp4    = (data['ph_isEM'] & LP4_MASK_BIT) == 0

        id_masks = {
            'TT': baseline &  is_tight &  is_iso & is_lp4,
            'TL': baseline &  is_tight & ~is_iso & is_lp4,
            'LT': baseline & ~is_tight &  is_iso & is_lp4,
            'LL': baseline & ~is_tight & ~is_iso & is_lp4,
        }

        truth_masks = {
            'JFP':   data['ph_truthJFP']   == 1,
            'Other': data['ph_truthother'] == 1,
        }

        region_masks = get_region_masks(data)

        for rt, rn in REGIONS:
            key      = f"{rt}/{rn}"
            reg_mask = region_masks[rt][rn]

            for id_cat, id_mask in id_masks.items():
                combined = reg_mask & id_mask

                for tc, tmask in truth_masks.items():
                    sel = combined & tmask
                    if not sel.any():
                        continue
                    w = totalweight[sel]

                    for var, cfg in VARIABLES.items():
                        raw = data[var][sel] * cfg['scale']
                        if cfg['norm'] is not None:
                            raw = raw / data[cfg['norm']][sel]
                        values = np.minimum(raw, cfg['bins'][-1])
                        h,  _ = np.histogram(values, bins=cfg['bins'], weights=w)
                        s2, _ = np.histogram(values, bins=cfg['bins'], weights=w ** 2)
                        hists[key][id_cat][proc][tc][var] += h
                        sumw2[key][id_cat][proc][tc][var] += s2

    return hists, sumw2

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(hists, key, id_cat, var, run_era_tag, lumi_label, stack_procs=None):
    if stack_procs is None:
        stack_procs = _STACK_PROCS
    stack_order = [(proc, tc) for tc in TRUTH_CATS for proc in stack_procs]

    cfg  = VARIABLES[var]
    bins = cfg['bins']
    bw   = np.diff(bins)

    fig, ax = plt.subplots(figsize=(7, 5))

    bottoms = np.zeros(len(bw))
    for proc, tc in stack_order:
        counts  = hists[key][id_cat][proc][tc][var]
        density = counts / bw
        ax.bar(
            bins[:-1], density, width=bw,
            bottom=bottoms, align='edge',
            label=f'{proc} ({tc})',
            color=PROCESS_COLORS[proc][tc],
            alpha=0.85,
        )
        bottoms += density

    ax.set_xlabel(cfg['label'], fontsize=16, labelpad=6, loc='right')
    ax.set_ylabel('Events / bin', fontsize=16, labelpad=6, loc='top')
    ax.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax.set_xlim(bins[0], bins[-1])
    ax.set_yscale('log')

    ymax = bottoms.max()
    if ymax > 0:
        _, ytop = ax.get_ylim()
        ybot    = ax.get_ylim()[0]
        log_range = np.log10(ytop) - np.log10(max(ybot, 1e-6))
        ax.set_ylim(top=10 ** (np.log10(ytop) + 0.55 * log_range))

    # ATLAS label
    t_atlas = ax.text(0.05, 0.97, 'ATLAS',
                      transform=ax.transAxes, fontsize=14, va='top',
                      fontweight='bold', fontstyle='italic', fontfamily='sans-serif')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb       = t_atlas.get_window_extent(renderer=renderer)
    x1_axes  = ax.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1_axes + 0.01, 0.97, 'Internal',
            transform=ax.transAxes, fontsize=14, va='top', fontfamily='sans-serif')

    ax.text(0.05, 0.90, lumi_label,
            transform=ax.transAxes, fontsize=11, va='top')
    ax.text(0.05, 0.84, REGION_LABELS[key],
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')
    ax.text(0.05, 0.78, ID_CATS[id_cat],
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')

    # Legend reversed so top-of-stack entry appears first
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              fontsize=9, framealpha=0.8, ncol=2,
              loc='upper right', bbox_to_anchor=(0.97, 0.97))

    fig.tight_layout()

    region_tag = key.replace('/', '_').replace('-', '_')
    fname = f"{region_tag}_{var}_{id_cat}_{run_era_tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Process-composition summary plots
# ---------------------------------------------------------------------------

def _get_yield(hists, key, id_cat, proc, truth_cat):
    """Total weighted yield for (region, id, proc, truth_cat). JFP+Other sums both."""
    if truth_cat == 'JFP+Other':
        return (hists[key][id_cat][proc]['JFP'][_YIELD_VAR].sum() +
                hists[key][id_cat][proc]['Other'][_YIELD_VAR].sum())
    return hists[key][id_cat][proc][truth_cat][_YIELD_VAR].sum()


def make_composition_plot(hists, id_cat, truth_cat, run_era_tag, lumi_label,
                          stack_procs=None):
    """Two-panel figure: absolute yield (left, log) and process fraction (right) per region.

    Shows which processes populate each region for a given photon ID category and
    truth category.
    """
    if stack_procs is None:
        stack_procs = _STACK_PROCS
    region_keys = [f"{rt}/{rn}" for rt, rn in REGIONS]
    # Short labels to fit below bars
    region_lbls = [REGION_LABELS[k].replace(' ', '\n') for k in region_keys]
    x     = np.arange(len(region_keys))
    width = 0.6

    # collect per-process yields across all regions
    color_key = 'Other' if truth_cat == 'Other' else 'JFP'
    yields = {proc: np.array([_get_yield(hists, k, id_cat, proc, truth_cat)
                               for k in region_keys])
              for proc in stack_procs}
    clipped = {proc: np.maximum(yields[proc], 0.) for proc in stack_procs}
    clip_totals = sum(clipped.values())
    totals = sum(yields.values())
    fracs  = {proc: np.where(clip_totals > 0, clipped[proc] / clip_totals, 0.)
              for proc in stack_procs}

    fig, (ax_abs, ax_frac) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, use_log, ylabel in [
        (ax_abs,  yields, True,  'Events'),
        (ax_frac, fracs,  False, 'Fraction'),
    ]:
        bottoms = np.zeros(len(region_keys))
        for proc in stack_procs:
            vals = data[proc]
            ax.bar(x, vals, width=width, bottom=bottoms,
                   label=proc, color=PROCESS_COLORS[proc][color_key], alpha=0.85)
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(region_lbls, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=13, loc='top')
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_xlim(-0.5, len(region_keys) - 0.5)
        ax.yaxis.grid(True, linestyle='--', color='lightgray', linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

        if use_log:
            ax.set_yscale('log')
            ymax = bottoms.max()
            if ymax > 0:
                _, ytop = ax.get_ylim()
                log_range = np.log10(ytop) - np.log10(ax.get_ylim()[0])
                ax.set_ylim(top=10 ** (np.log10(ytop) + 0.45 * log_range))
        else:
            ax.set_ylim(0, 1.35)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

        # ATLAS label
        t_atlas = ax.text(0.05, 0.97, 'ATLAS',
                          transform=ax.transAxes, fontsize=12, va='top',
                          fontweight='bold', fontstyle='italic')
        fig.canvas.draw()
        bb      = t_atlas.get_window_extent(renderer=fig.canvas.get_renderer())
        x1_axes = ax.transAxes.inverted().transform((bb.x1, bb.y0))[0]
        ax.text(x1_axes + 0.01, 0.97, 'Internal',
                transform=ax.transAxes, fontsize=12, va='top')
        ax.text(0.05, 0.91, lumi_label,
                transform=ax.transAxes, fontsize=9, va='top')
        ax.text(0.05, 0.85, ID_CATS[id_cat],
                transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace')
        ax.text(0.05, 0.79, f'{truth_cat} photons',
                transform=ax.transAxes, fontsize=9, va='top')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1],
                  fontsize=9, framealpha=0.8, loc='upper right',
                  bbox_to_anchor=(0.97, 0.97))

    fig.tight_layout()

    tc_tag = truth_cat.replace('+', 'p')
    fname  = f"composition_{id_cat}_{tc_tag}_{run_era_tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='JFP/Other process composition plots.'
    )
    parser.add_argument(
        '--include-other-mc', action='store_true',
        help=(
            'Include all remaining background MC files (ttbar, single-top, diboson, etc.) '
            'summed into an "OtherMC" category shown in gray.'
        ),
    )
    args = parser.parse_args()

    stack_procs = _STACK_PROCS + ([_OTHER_MC] if args.include_other_mc else [])

    for run2, run_era_tag, lumi_label in [
        (True,  'Run2', r'$\sqrt{s}=13\ \mathrm{TeV},\ 140\ \mathrm{fb}^{-1}$'),
        (False, 'Run3', r'$\sqrt{s}=13.6\ \mathrm{TeV},\ 160\ \mathrm{fb}^{-1}$'),
    ]:
        hists, sumw2 = accumulate(run2, args.include_other_mc)

        n_plots = 0
        for rt, rn in REGIONS:
            key = f"{rt}/{rn}"
            for id_cat in ID_CATS:
                for var in VARIABLES:
                    make_plot(hists, key, id_cat, var, run_era_tag, lumi_label,
                              stack_procs=stack_procs)
                    n_plots += 1

        for id_cat in ID_CATS:
            for tc in TRUTH_COMP_CATS:
                make_composition_plot(hists, id_cat, tc, run_era_tag, lumi_label,
                                      stack_procs=stack_procs)
                n_plots += 1

        era = 'Run 2' if run2 else 'Run 3'
        print(f"Saved {n_plots} plots for {era} to {OUTPUT_DIR}/")
