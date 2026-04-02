#!/usr/bin/env python3
"""
Truth category composition (Prompt / EFP / JFP / Other) for Tight photons
across analysis regions and MC processes.

One stacked-bar plot per (era, process): x = region, y = fraction of
weighted yield in each truth category.

Output: truth_composition_plots/<era>_<process>.{pdf,png}
"""

import glob
import os

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Nimbus Sans'
import matplotlib.pyplot as plt
import numpy as np
import uproot

from abcd_utils import get_region_masks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = '/data/mhance/SUSY/ntuples/v4.2'
OUTPUT_DIR = 'truth_composition_plots'

REGIONS = [
    ('Preselection', '0L',              'Presel'),
    ('VR',           '0L-mT-mid',       'VR-mid'),
    ('SR',           '0L-mT-low-loose', 'SR-low'),
    ('SR',           '0L-mT-mid-loose', 'SR-mid'),
    ('SR',           '0L-mT-hgh-loose', 'SR-high'),
]

PROCESSES = [
    ('Znunu',     r'$Z(\nu\nu)$+jets',  ['Znunu']),
    ('Wtaunu',    r'$W(\tau\nu)$+jets', ['Wtaunu']),
    ('nunugamma', r'$\nu\nu\gamma$',    ['nunugamma']),
]
PROC_TAGS = [t for t, _, _ in PROCESSES]

TRUTH_CATS = ['Prompt', 'EFP', 'JFP', 'Other']

TRUTH_COLORS = {
    'Prompt': '#377eb8',
    'EFP':    '#ff7f00',
    'JFP':    '#e41a1c',
    'Other':  '#999999',
}

NEEDED_BRANCHES = [
    'ph_select_baseline', 'ph_select_tightID',
    'ph_truthprompt', 'ph_truthEFP', 'ph_truthJFP', 'ph_truthother',
    'weight_total', 'weight_fjvt_effSF', 'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
    'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nPhotons_baseline', 'nPhotons_skims', 'nPhotons_baseline_noOR',
    'mTGammaMet', 'dPhiGammaMet', 'nTau20_veryloose', 'met_signif', 'ph_pt',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

_SKIP = ('gammajet', 'jetjet', '_jj_', 'N2_', 'signal', 'data_')

def is_background_mc(fname):
    return not any(s in fname for s in _SKIP)

def proc_tags_for_file(fname):
    return [tag for tag, _, substrings in PROCESSES if any(s in fname for s in substrings)]

# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(run2):
    era_tag = 'Run2' if run2 else 'Run3'
    mc_tag  = 'mc20' if run2 else 'mc23'

    # counts[proc][region_key][truth_cat] = sum of weights
    counts = {
        proc: {f'{reg}/{sub}': {tc: 0. for tc in TRUTH_CATS} for reg, sub, _ in REGIONS}
        for proc in PROC_TAGS
    }

    all_files = sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root')))
    bkg_files = [f for f in all_files
                 if mc_tag in os.path.basename(f)
                 and is_background_mc(os.path.basename(f))]

    print(f'\nAccumulating {era_tag}: {len(bkg_files)} background MC files')

    for i, fp in enumerate(bkg_files, 1):
        fname = os.path.basename(fp)
        tags  = proc_tags_for_file(fname)
        if not tags:
            continue
        print(f'  [{i:3d}/{len(bkg_files)}] {fname}')
        try:
            with uproot.open(fp) as uf:
                if 'picontuple' not in uf:
                    print('    (no picontuple, skipping)')
                    continue
                data = uf['picontuple'].arrays(NEEDED_BRANCHES, library='np')
        except Exception as exc:
            print(f'    ERROR: {exc}')
            continue

        w = (data['weight_total'] *
             data['weight_fjvt_effSF'] *
             data['weight_ftag_effSF_GN2v01_Continuous'] *
             data['weight_jvt_effSF'])

        region_masks = get_region_masks(data)
        tight = (data['ph_select_baseline'] == 1) & (data['ph_select_tightID'] == 1)

        truth_masks = {
            'Prompt': data['ph_truthprompt'] == 1,
            'EFP':    data['ph_truthEFP']    == 1,
            'JFP':    data['ph_truthJFP']    == 1,
            'Other':  data['ph_truthother']  == 1,
        }

        for reg, sub, _ in REGIONS:
            region_key = f'{reg}/{sub}'
            try:
                reg_mask = region_masks[reg][sub]
            except KeyError:
                continue
            base = reg_mask & tight
            for tc, truth_mask in truth_masks.items():
                sel = base & truth_mask
                sw  = float(w[sel].sum())
                for proc_tag in tags:
                    counts[proc_tag][region_key][tc] += sw

    return counts

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(counts, proc_tag, proc_label, era_tag, lumi_label):
    region_keys   = [f'{reg}/{sub}' for reg, sub, _ in REGIONS]
    region_labels = [label for _, _, label in REGIONS]
    x = np.arange(len(REGIONS))

    fig, ax = plt.subplots(figsize=(7, 5))

    totals = np.array([sum(counts[proc_tag][rk].values()) for rk in region_keys])

    bottoms = np.zeros(len(REGIONS))
    handles = []
    for tc in TRUTH_CATS:
        fracs = np.array([
            counts[proc_tag][rk][tc] / totals[i] if totals[i] > 0 else 0.
            for i, rk in enumerate(region_keys)
        ])
        bars = ax.bar(x, fracs, bottom=bottoms, width=0.6,
                      color=TRUTH_COLORS[tc], label=tc, alpha=0.9)
        handles.append(bars)
        bottoms += fracs

    # Annotate each bar with total weighted yield
    for i, total in enumerate(totals):
        if total <= 0:
            continue
        ax.text(x[i], 0.1, f'{total:.2g}', ha='center', va='center',
                fontsize=9, color='black', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, fontsize=12)
    ax.set_ylabel('Fraction of Tight photons', fontsize=16, labelpad=6, loc='top')
    ax.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax.tick_params(axis='x', which='both', bottom=False)
    ax.set_ylim(0, 1)

    # ATLAS label
    t_atlas = ax.text(0.05, 0.97, 'ATLAS',
                      transform=ax.transAxes, fontsize=14, va='top',
                      fontweight='bold', fontstyle='italic', fontfamily='Nimbus Sans')
    fig.canvas.draw()
    bb      = t_atlas.get_window_extent(renderer=fig.canvas.get_renderer())
    x1_axes = ax.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1_axes + 0.01, 0.97, 'Internal',
            transform=ax.transAxes, fontsize=14, va='top', fontfamily='Nimbus Sans')
    ax.text(0.05, 0.90, lumi_label,
            transform=ax.transAxes, fontsize=11, va='top')
    ax.text(0.05, 0.84, 'Tight photons',
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')
    ax.text(0.05, 0.78, proc_label,
            transform=ax.transAxes, fontsize=11, va='top')

    # Legend: reverse so top of stack is first
    lhandles, llabels = ax.get_legend_handles_labels()
    ax.legend(lhandles[::-1], llabels[::-1], fontsize=11, framealpha=0.8,
              loc='upper right', bbox_to_anchor=(0.97, 0.97))

    fname = f'{era_tag}_{proc_tag}'
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for run2, era_tag, lumi_label in [
        (True,  'Run2', r'$\sqrt{s}=13\ \mathrm{TeV},\ 140\ \mathrm{fb}^{-1}$'),
        (False, 'Run3', r'$\sqrt{s}=13.6\ \mathrm{TeV},\ 160\ \mathrm{fb}^{-1}$'),
    ]:
        counts = accumulate(run2)
        for proc_tag, proc_label, _ in PROCESSES:
            make_plot(counts, proc_tag, proc_label, era_tag, lumi_label)
