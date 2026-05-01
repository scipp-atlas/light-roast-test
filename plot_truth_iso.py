#!/usr/bin/env python3
"""
Isolation distributions for photons in pT slices, overlaying
different truth categories (Prompt, EFP, JFP, Other) on a log-y scale.
Distributions are normalized to unit area for shape comparison.

Selection: baseline + Tight ID photons in Preselection / 0L.

Matrix of plots:
  {Run 2, Run 3} × {Wtaunu, Znunu, inclusive}
  × {10–15, 15–25, 25–40, >40 GeV pT}
  × {topoetcone40/pT, topoetcone20/pT, ptcone20/pT}

Output: truth_iso_plots/<era>_<process>_pt<ptbin>_<var>.{pdf,png}
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
OUTPUT_DIR = 'truth_iso_plots'

VARIABLES = ['ph_topoetcone40', 'ph_topoetcone20', 'ph_ptcone20', 'ph_pt_over_truthpt']

VAR_LABELS = {
    'ph_topoetcone40':    r'$(\mathrm{topo}E_{\rm T}^{\rm cone40} - 2.45\ \mathrm{GeV})\ /\ p_{\rm T}^{\gamma}$',
    'ph_topoetcone20':    r'$\mathrm{topo}E_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
    'ph_ptcone20':        r'$p_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
    'ph_pt_over_truthpt': r'$p_{\rm T}^{\gamma,\,\mathrm{reco}}\ /\ p_{\rm T}^{\gamma,\,\mathrm{truth}}$',
}

# Vertical cut lines — only for isolation variables
ISO_CUT = {
    'ph_topoetcone40': 0.022,
    'ph_topoetcone20': 0.065,
    'ph_ptcone20':     0.05,
}

# Upper x-axis limit (last visible bin acts as overflow)
VAR_XMAX = {
    'ph_topoetcone40':    1.0,
    'ph_topoetcone20':    1.0,
    'ph_ptcone20':        1.0,
    'ph_pt_over_truthpt': 8.0,
}

def _var_bins(*segments):
    """Build bin edges from (lo, hi, step) segments with progressively coarser spacing."""
    parts = []
    for lo, hi, step in segments:
        edges = np.arange(lo, hi + step * 0.5, step)
        edges = edges[edges <= hi + 1e-9]
        if parts:
            edges = edges[1:]
        parts.append(edges)
    return np.concatenate(parts)

BINS = {
    'ph_topoetcone40': _var_bins(
        (-0.20, 0.30, 0.10),
        ( 0.30, 1.00, 0.20),
        ( 1.00, 2.00, 0.40),
        ( 2.00, 5.00, 1.00),
    ),
    'ph_topoetcone20': _var_bins(
        (-0.20, 0.50, 0.10),
        ( 0.50, 1.00, 0.20),
        ( 1.00, 2.00, 0.40),
        ( 2.00, 5.00, 1.00),
    ),
    'ph_ptcone20': _var_bins(
        ( 0.00, 0.30, 0.10),
        ( 0.30, 1.00, 0.20),
        ( 1.00, 2.00, 0.40),
        ( 2.00, 5.00, 1.00),
    ),
    'ph_pt_over_truthpt': _var_bins(
        ( 0.00, 8.00, 0.10),
        ( 8.00,10.00, 0.20),
    ),
}

# Processes: tag, display label, filename substrings (None = all bkg MC)
PROCESSES = [
    ('Znunu',     r'$Z(\nu\nu)$+jets',        ['Znunu']),
    ('Wtaunu',    r'$W(\tau\nu)$+jets',        ['Wtaunu']),
    ('nunugamma', r'$\nu\nu\gamma$',           ['nunugamma']),
]
PROC_TAGS = [tag for tag, _, _ in PROCESSES]

# Truth categories
TRUTH_CATS = ['Prompt', 'EFP', 'JFP', 'Other']

TRUTH_STYLES = {
    'Prompt': {'color': '#377eb8', 'marker': 'o', 'label': 'Prompt'},
    'EFP':    {'color': '#ff7f00', 'marker': 's', 'label': 'EFP'},
    'JFP':    {'color': '#e41a1c', 'marker': '^', 'label': 'JFP'},
    'Other':  {'color': '#999999', 'marker': 'D', 'label': 'Other'},
}

_NTUPLE_VARS = ['ph_topoetcone40', 'ph_topoetcone20', 'ph_ptcone20']  # direct branches only

NEEDED_BRANCHES = _NTUPLE_VARS + [
    'ph_pt', 'ph_truthpt', 'ph_select_baseline', 'ph_select_tightID',
    'ph_truthprompt', 'ph_truthEFP', 'ph_truthJFP', 'ph_truthother',
    'weight_total', 'weight_fjvt_effSF', 'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
    'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nPhotons_baseline', 'nPhotons_skims', 'nPhotons_baseline_noOR',
    'mTGammaMet', 'dPhiGammaMet', 'nTau20_veryloose', 'met_signif',
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def file_era(fname):
    if 'mc20' in fname: return 'Run2'
    if 'mc23' in fname: return 'Run3'
    return None

_SKIP = ('gammajet', 'jetjet', '_jj_', 'N2_', 'signal', 'data_')

def is_background_mc(fname):
    return not any(s in fname for s in _SKIP)

def proc_tags_for_file(fname):
    tags = []
    for tag, _, substrings in PROCESSES:
        if substrings is None:
            tags.append(tag)
        elif any(s in fname for s in substrings):
            tags.append(tag)
    return tags

# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(run2):
    """Fill histograms for all (proc, truth_cat, var).

    run2=True: Run 2 only (mc20).
    run2=False: Run 3 only (mc23).
    run2=None: Run 2+3 combined.
    """
    if run2 is True:
        era_tag  = 'Run2'
        mc_tags  = ('mc20',)
    elif run2 is False:
        era_tag  = 'Run3'
        mc_tags  = ('mc23',)
    else:
        era_tag  = 'Run2p3'
        mc_tags  = ('mc20', 'mc23')

    def _zeros(var):
        return np.zeros(len(BINS[var]) - 1)

    hists = {
        proc: {tc: {var: _zeros(var) for var in VARIABLES} for tc in TRUTH_CATS}
        for proc in PROC_TAGS
    }
    sumw2 = {
        proc: {tc: {var: _zeros(var) for var in VARIABLES} for tc in TRUTH_CATS}
        for proc in PROC_TAGS
    }

    all_files = sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root')))
    bkg_files = [f for f in all_files
                 if any(t in os.path.basename(f) for t in mc_tags)
                 and is_background_mc(os.path.basename(f))]

    print(f'\nAccumulating {era_tag}: {len(bkg_files)} background MC files ({", ".join(mc_tags)})')

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

        region_mask = get_region_masks(data)['Preselection']['0L']
        baseline    = (data['ph_select_baseline'] == 1) & (data['ph_select_tightID'] == 1)

        truth_masks = {
            'Prompt': data['ph_truthprompt'] == 1,
            'EFP':    data['ph_truthEFP']    == 1,
            'JFP':    data['ph_truthJFP']    == 1,
            'Other':  data['ph_truthother']  == 1,
        }

        base = region_mask & baseline
        for tc, truth_mask in truth_masks.items():
            sel = base & truth_mask
            if not sel.any():
                continue
            w_sel = w[sel]
            for var in VARIABLES:
                # Compute variable values and per-event validity mask
                if var == 'ph_topoetcone40':
                    raw   = (data[var][sel] - 2450.) / data['ph_pt'][sel]
                    valid = np.ones(raw.shape, dtype=bool)
                elif var in ('ph_topoetcone20', 'ph_ptcone20'):
                    raw   = data[var][sel] / data['ph_pt'][sel]
                    valid = np.ones(raw.shape, dtype=bool)
                elif var == 'ph_pt_over_truthpt':
                    truth_pt = data['ph_truthpt'][sel]
                    valid    = truth_pt > 0
                    raw      = np.where(valid, data['ph_pt'][sel] / np.where(valid, truth_pt, 1.), 0.)
                else:
                    raw   = data[var][sel]
                    valid = np.ones(raw.shape, dtype=bool)

                values = np.minimum(raw[valid], BINS[var][-1])
                wv     = w_sel[valid]
                h,  _  = np.histogram(values, bins=BINS[var], weights=wv)
                s2, _  = np.histogram(values, bins=BINS[var], weights=wv ** 2)
                for proc_tag in tags:
                    hists[proc_tag][tc][var] += h
                    sumw2[proc_tag][tc][var] += s2

    return hists, sumw2

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(hists, sumw2, proc_tag, proc_label, var,
              era_tag, lumi_label):
    bin_edges   = BINS[var]
    bin_widths  = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Number of visible bins: those with right edge <= x_max.
    # All content beyond that is merged into the last visible bin (overflow).
    x_max  = VAR_XMAX[var]
    n_vis  = int(np.searchsorted(bin_edges, x_max, side='left'))  # index of x_max edge
    bw_vis = bin_widths[:n_vis]
    bc_vis = bin_centers[:n_vis]

    ymax = 0.
    for tc in TRUTH_CATS:
        style = TRUTH_STYLES[tc]
        h  = hists[proc_tag][tc][var]
        s2 = sumw2[proc_tag][tc][var]

        # Merge everything beyond the visible range into the last visible bin
        h_plot       = h[:n_vis].copy()
        h_plot[-1]  += h[n_vis:].sum()
        s2_plot      = s2[:n_vis].copy()
        s2_plot[-1] += s2[n_vis:].sum()

        total = (h_plot * bw_vis).sum()
        if total <= 0:
            continue

        den  = (h_plot  / bw_vis) / total
        derr = (np.sqrt(s2_plot) / bw_vis) / total
        mask = den > 0

        if not mask.any():
            continue

        ax.errorbar(
            bc_vis[mask], den[mask], yerr=derr[mask],
            fmt=style['marker'], color=style['color'], label=style['label'],
            markersize=4, linewidth=0, elinewidth=1, capsize=2,
        )
        ymax = max(ymax, den[mask].max())

    ax.set_xlabel(VAR_LABELS[var], fontsize=16, labelpad=6, loc='right')
    ax.set_ylabel('Normalized yield / bin width', fontsize=16, labelpad=6, loc='top')
    ax.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax.set_xlim(bin_edges[0], x_max)
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-1)
    if ymax > 0:
        _, ytop = ax.get_ylim()
        log_range = np.log10(ytop) - np.log10(1e-1)
        ax.set_ylim(top=10 ** (np.log10(ytop) + 0.45 * log_range))

    if var in ISO_CUT:
        ax.axvline(ISO_CUT[var], color='k', linestyle=':', linewidth=1.2)

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
    ax.text(0.05, 0.84, 'Preselection / 0L  —  Tight',
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')
    ax.text(0.05, 0.78, proc_label,
            transform=ax.transAxes, fontsize=11, va='top')

    ax.legend(fontsize=12, framealpha=0.8, loc='upper right', bbox_to_anchor=(0.97, 0.97))

    fname = f'{era_tag}_{proc_tag}_{var}'
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for run2, lumi_label in [
        (True,  r'$\sqrt{s}=13\ \mathrm{TeV},\ 140\ \mathrm{fb}^{-1}$'),
        (False, r'$\sqrt{s}=13.6\ \mathrm{TeV},\ 160\ \mathrm{fb}^{-1}$'),
        (None,  r'$\sqrt{s}=13\text{--}13.6\ \mathrm{TeV},\ \mathrm{Run\ 2+3\ combined}$'),
    ]:
        hists, sumw2 = accumulate(run2)
        era_tag = 'Run2' if run2 is True else ('Run3' if run2 is False else 'Run2p3')

        n_plots = 0
        for proc_tag, proc_label, _ in PROCESSES:
            for var in VARIABLES:
                make_plot(hists, sumw2, proc_tag, proc_label, var,
                          era_tag, lumi_label)
                n_plots += 1

        print(f'Saved {n_plots} plots for {era_tag} to {OUTPUT_DIR}/')
