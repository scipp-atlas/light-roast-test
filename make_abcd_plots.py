#!/usr/bin/env python3
"""
Make ABCD isolation variable plots.

Produces weighted stacked histograms of ph_topoetcone40, ph_topoetcone20, and
ph_ptcone20 for Tight / LoosePrime4 / Loose photon ID criteria, split by MC
truth category (Real, EFP, JFP, Other), for two event selections
(Preselection-0L and VR-0L-mT-mid) and two run eras (Run 2 / Run 3).

Output: abcd_iso_plots/<selection>_<variable>_<ID>_<RunEra>.{pdf,png}
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Nimbus Sans'

import matplotlib.pyplot as plt
import numpy as np
import uproot
import os
import glob

from abcd_utils import (
    get_region_masks,
    get_photon_id_masks,
    get_truth_masks,
    fill_iso_histograms,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = "/data/mhance/SUSY/ntuples/v4"
OUTPUT_DIR = "abcd_iso_plots"

VARIABLES = ['ph_topoetcone40', 'ph_topoetcone20', 'ph_ptcone20']

VAR_LABELS = {
    'ph_topoetcone40': r'$\mathrm{topo}E_{\rm T}^{\rm cone40}\ /\ p_{\rm T}^{\gamma}$',
    'ph_topoetcone20': r'$\mathrm{topo}E_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
    'ph_ptcone20':     r'$p_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
}

# Bin edges as dimensionless iso/pT ratios
BINS = {
    'ph_topoetcone40': np.arange(-0.20, 5.01, 0.1),  # -0.2 to 5.0, 0.05-wide bins
    'ph_topoetcone20': np.arange(-0.20, 5.01, 0.1),  # -0.2 to 5.0, 0.05-wide bins
    'ph_ptcone20':     np.arange( 0.00, 5.01, 0.1),  #  0.0 to 5.0, 0.05-wide bins
}

SELECTIONS = [
    ('Preselection', '0L'),
    ('VR',           '0L-mT-mid'),
]

SEL_LABELS = {
    'Preselection-0L':    'Preselection-0L',
    'VR-0L-mT-mid':       'VR-0L-mT-mid',
}

ID_CRITERIA = ['Tight', 'LoosePrime4', 'Loose']

ID_LABELS = {
    'Tight':       'Tight',
    'LoosePrime4': "LoosePrime4 (not Tight)",
    'Loose':       'Loose (not Tight)',
}

TRUTH_CATS   = ['Real', 'EFP', 'JFP', 'Other']
TRUTH_COLORS = {
    'Real':  '#377eb8',
    'EFP':   '#ff7f00',
    'JFP':   '#e41a1c',
    'Other': '#999999',
}

# Branches needed from the ntuple
NEEDED_BRANCHES = (
    list(VARIABLES) + [
        'ph_select_baseline', 'ph_select_tightID', 'ph_isEM',
        'ph_truthprompt', 'ph_truthJFP', 'ph_truthEFP', 'ph_truthother',
        'weight_total', 'weight_fjvt_effSF',
        'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
        'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'ph_pt',
        'mindPhiJetMet', 'nBTagJets', 'nElectrons', 'nMuons',
        'mTGammaMet', 'dPhiGammaMet', 'nTau20_veryloose', 'met_signif',
    ]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: decide whether a file belongs to background MC for a given run era
# ---------------------------------------------------------------------------

def is_background_mc(filepath, run2):
    fname = os.path.basename(filepath)
    for skip in ('gammajet', 'jetjet', '_jj_', 'N2_', 'signal', 'data_'):
        if skip in fname:
            return False
    is_run2 = 'mc20' in fname
    is_run3 = 'mc23' in fname
    if run2  and not is_run2: return False
    if not run2 and not is_run3: return False
    return True

# ---------------------------------------------------------------------------
# Accumulate histograms from all background MC files
# ---------------------------------------------------------------------------

def accumulate_histograms(run2):
    """
    Loop over all background MC ntuples for the given run era and accumulate
    weighted histograms.

    Returns:
        hists[sel_key][id_crit][variable][truth_cat]  -> summed-weight array
        sumw2[sel_key][id_crit][variable][truth_cat]  -> sum-of-weights-squared array
    """
    def _zero_structure():
        return {
            f'{st}-{sn}': {
                id_crit: {
                    var: {cat: np.zeros(len(BINS[var]) - 1) for cat in TRUTH_CATS}
                    for var in VARIABLES
                }
                for id_crit in ID_CRITERIA
            }
            for st, sn in SELECTIONS
        }

    hists = _zero_structure()
    sumw2 = _zero_structure()

    all_files = sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root')))
    bkg_files = [f for f in all_files if is_background_mc(f, run2)]

    era = 'Run 2' if run2 else 'Run 3'
    print(f"\nAccumulating {era}: {len(bkg_files)} background MC files")

    for i, fp in enumerate(bkg_files, 1):
        print(f"  [{i:3d}/{len(bkg_files)}] {os.path.basename(fp)}")
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

        region_masks = get_region_masks(data)
        id_masks     = get_photon_id_masks(data)

        for sel_type, sel_name in SELECTIONS:
            sel_key    = f'{sel_type}-{sel_name}'
            event_mask = region_masks[sel_type][sel_name]

            for id_crit, id_mask in id_masks.items():
                for var in VARIABLES:
                    cat_counts, cat_sumw2 = fill_iso_histograms(
                        data, event_mask, id_mask, var,
                        BINS[var], totalweight, norm_variable='ph_pt'
                    )
                    for cat in TRUTH_CATS:
                        hists[sel_key][id_crit][var][cat] += cat_counts[cat]
                        sumw2[sel_key][id_crit][var][cat] += cat_sumw2[cat]

    return hists, sumw2

# ---------------------------------------------------------------------------
# Make and save a single plot
# ---------------------------------------------------------------------------

def make_plot(hists, sel_key, id_crit, var, run_era_label, run_era_tag, lumi_label):
    bin_edges  = BINS[var]
    bin_widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Stacked filled histogram
    bottoms = np.zeros(len(bin_widths))
    for cat in TRUTH_CATS:
        density = hists[sel_key][id_crit][var][cat] / bin_widths   # events / GeV
        ax.bar(
            bin_edges[:-1], density, width=bin_widths,
            bottom=bottoms, align='edge',
            label=cat, color=TRUTH_COLORS[cat], alpha=0.85,
        )
        bottoms += density

    # Axis formatting
    ax.set_xlabel(VAR_LABELS[var], fontsize=16, labelpad=6, loc='right')
    ax.set_ylabel('Events / bin',  fontsize=16, labelpad=6, loc='top')
    ax.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_yscale('log')

    # Extra headroom above distribution so labels don't overlap.
    # On log scale, extend the top by 40% of the current log range.
    ymax = bottoms.max()
    if ymax > 0:
        ybot, ytop = ax.get_ylim()
        log_range = np.log10(ytop) - np.log10(max(ybot, 1e-6))
        ax.set_ylim(top=10 ** (np.log10(ytop) + 0.35 * log_range))

    # ATLAS label: "ATLAS" bold italic Helvetica, "Internal" regular — placed side by side
    t_atlas = ax.text(0.05, 0.97, 'ATLAS',
                      transform=ax.transAxes, fontsize=14, va='top',
                      fontweight='bold', fontstyle='italic', fontfamily='Nimbus Sans')
    fig.canvas.draw()   # needed to compute rendered text width
    renderer = fig.canvas.get_renderer()
    bb = t_atlas.get_window_extent(renderer=renderer)
    x1_axes = ax.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1_axes + 0.01, 0.97, 'Internal',
            transform=ax.transAxes, fontsize=14, va='top', fontfamily='Nimbus Sans')

    ax.text(0.05, 0.90, lumi_label,
            transform=ax.transAxes, fontsize=11, va='top')
    ax.text(0.05, 0.83, SEL_LABELS[sel_key],
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')
    ax.text(0.05, 0.77, ID_LABELS[id_crit],
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')

    # Legend: reverse handles so order matches the visual stack (top category first)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=14, framealpha=0.8,
              loc='upper right', bbox_to_anchor=(0.97, 0.97))

    fig.tight_layout()

    fname = f"{sel_key}_{var}_{id_crit}_{run_era_tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(
            os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
            dpi=300, bbox_inches='tight',
        )
    plt.close(fig)

# ---------------------------------------------------------------------------
# JFP comparison plot: Tight vs LoosePrime4 vs Loose, markers with error bars
# ---------------------------------------------------------------------------

ID_STYLES = {
    'Tight':       {'color': '#1a237e', 'marker': 'o', 'label': 'Tight'},
    'LoosePrime4': {'color': '#b71c1c', 'marker': 's', 'label': 'LoosePrime4 (not Tight)'},
    'Loose':       {'color': '#1b5e20', 'marker': '^', 'label': 'Loose (not Tight)'},
}

def make_jfp_comparison_plot(hists, sumw2, sel_key, var, run_era_label, run_era_tag, lumi_label):
    bin_edges   = BINS[var]
    bin_widths  = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.06)
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    def jfp_counts(h, s2, id_crit):
        """JFP + Other counts and sumw2, combined."""
        counts = h[sel_key][id_crit][var]['JFP'] + h[sel_key][id_crit][var]['Other']
        sw2    = s2[sel_key][id_crit][var]['JFP'] + s2[sel_key][id_crit][var]['Other']
        return counts, sw2

    # Pre-compute Tight density for ratio panel
    tight_c, tight_s2 = jfp_counts(hists, sumw2, 'Tight')
    tight_density = tight_c / bin_widths
    tight_errs    = np.sqrt(tight_s2) / bin_widths

    # --- Main panel ---
    ymax = 0.
    for id_crit, style in ID_STYLES.items():
        c, s2         = jfp_counts(hists, sumw2, id_crit)
        density      = c / bin_widths
        density_errs = np.sqrt(s2) / bin_widths
        mask = density > 0
        ax_main.errorbar(
            bin_centers[mask], density[mask], yerr=density_errs[mask],
            fmt=style['marker'], color=style['color'], label=style['label'],
            markersize=4, linewidth=0, elinewidth=1, capsize=2,
        )
        ymax = max(ymax, density[mask].max() if mask.any() else 0)

    ax_main.set_ylabel('Events / bin', fontsize=16, labelpad=6, loc='top')
    ax_main.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.set_xlim(bin_edges[0], bin_edges[-1])
    ax_main.set_yscale('log')
    ax_main.set_ylim(bottom=1e-2)
    if ymax > 0:
        _, ytop = ax_main.get_ylim()
        log_range = np.log10(ytop) - np.log10(1e-2)
        ax_main.set_ylim(top=10 ** (np.log10(ytop) + 0.3 * log_range))

    # ATLAS label
    t_atlas = ax_main.text(0.05, 0.97, 'ATLAS',
                           transform=ax_main.transAxes, fontsize=14, va='top',
                           fontweight='bold', fontstyle='italic', fontfamily='Nimbus Sans')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = t_atlas.get_window_extent(renderer=renderer)
    x1_axes = ax_main.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax_main.text(x1_axes + 0.01, 0.97, 'Internal',
                 transform=ax_main.transAxes, fontsize=14, va='top', fontfamily='Nimbus Sans')
    ax_main.text(0.05, 0.90, lumi_label,         transform=ax_main.transAxes, fontsize=11, va='top')
    ax_main.text(0.05, 0.83, SEL_LABELS[sel_key], transform=ax_main.transAxes, fontsize=11, va='top', fontfamily='monospace')
    ax_main.legend(fontsize=14, framealpha=0.8, loc='upper right', bbox_to_anchor=(0.97, 0.97))

    # --- Ratio panel ---
    ax_ratio.axhline(1., color='black', linewidth=0.8, linestyle='--')

    for id_crit, style in ID_STYLES.items():
        if id_crit == 'Tight':
            continue
        c, s2        = jfp_counts(hists, sumw2, id_crit)
        density      = c / bin_widths
        density_errs = np.sqrt(s2) / bin_widths

        valid = (tight_density > 0) & (density > 0)
        safe_tight = np.where(valid, tight_density, 1.)
        safe_dens  = np.where(valid, density,       1.)
        ratio      = np.where(valid, density / safe_tight, np.nan)
        ratio_errs = np.where(valid,
                               ratio * np.sqrt((density_errs / safe_dens) ** 2 +
                                               (tight_errs   / safe_tight) ** 2),
                               np.nan)
        ax_ratio.errorbar(
            bin_centers[valid], ratio[valid], yerr=ratio_errs[valid],
            fmt=style['marker'], color=style['color'],
            markersize=4, linewidth=0, elinewidth=1, capsize=2,
        )

    ax_ratio.set_xlabel(VAR_LABELS[var], fontsize=16, labelpad=6, loc='right')
    ax_ratio.set_ylabel('Ratio to Tight', fontsize=16, labelpad=6)
    ax_ratio.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax_ratio.set_ylim(0, 6)
    ax_ratio.yaxis.grid(True, color='lightgray', linewidth=0.7, zorder=0)
    ax_ratio.set_axisbelow(True)
    ax_ratio.axhline(1., color='black', linewidth=0.8, linestyle='--')  # redraw on top


    fname = f"jfp_comparison_{sel_key}_{var}_{run_era_tag}"
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for run2, run_era_label, run_era_tag, lumi_label in [
        (True,  'Run 2', 'Run2', r'$\sqrt{s}=13\ \mathrm{TeV},\ 140\ \mathrm{fb}^{-1}$'),
        (False, 'Run 3', 'Run3', r'$\sqrt{s}=13.6\ \mathrm{TeV},\ 160\ \mathrm{fb}^{-1}$'),
    ]:
        hists, sumw2 = accumulate_histograms(run2)

        n_plots = 0
        for sel_type, sel_name in SELECTIONS:
            sel_key = f'{sel_type}-{sel_name}'
            for id_crit in ID_CRITERIA:
                for var in VARIABLES:
                    make_plot(hists, sel_key, id_crit, var, run_era_label, run_era_tag, lumi_label)
                    n_plots += 1
            for var in VARIABLES:
                make_jfp_comparison_plot(hists, sumw2, sel_key, var, run_era_label, run_era_tag, lumi_label)
                n_plots += 1

        print(f"Saved {n_plots} plots for {run_era_label} to {OUTPUT_DIR}/")
