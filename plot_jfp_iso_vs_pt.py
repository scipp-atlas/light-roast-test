#!/usr/bin/env python3
"""
Isolation distributions for JFP photons in pT slices, overlaying
Tight / LoosePrime4 / Loose ID working points on a log-y scale
with a ratio-to-Tight panel below.

Matrix of plots:
  {Run 2, Run 3} × {Wtaunu, Znunu, inclusive}
  × {10–20, 20–40, >40 GeV pT}
  × {topoetcone40/pT, topoetcone20/pT, ptcone20/pT}

Region: Preselection / 0L
Output: jfp_iso_pt_plots/<era>_<process>_pt<ptbin>_<var>.{pdf,png}
"""

import glob
import os

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Nimbus Sans'
import matplotlib.pyplot as plt
import numpy as np
import uproot

from abcd_utils import get_region_masks, LoosePrimeDefs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = '/data/mhance/SUSY/ntuples/v4.2'
OUTPUT_DIR = 'jfp_iso_pt_plots'

VARIABLES = ['ph_topoetcone40', 'ph_topoetcone20', 'ph_ptcone20']

VAR_LABELS = {
    'ph_topoetcone40': r'$(\mathrm{topo}E_{\rm T}^{\rm cone40} - 2.45\ \mathrm{GeV})\ /\ p_{\rm T}^{\gamma}$',
    'ph_topoetcone20': r'$\mathrm{topo}E_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
    'ph_ptcone20':     r'$p_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
}

# Vertical lines marking the isolation cut for each variable
ISO_CUT = {
    'ph_topoetcone40': 0.022,
    'ph_topoetcone20': 0.065,
    'ph_ptcone20':     0.05,
}

def _var_bins(*segments):
    """Build bin edges from (lo, hi, step) segments with progressively coarser spacing."""
    parts = []
    for lo, hi, step in segments:
        edges = np.arange(lo, hi + step * 0.5, step)
        edges = edges[edges <= hi + 1e-9]
        if parts:
            edges = edges[1:]   # drop duplicate boundary point
        parts.append(edges)
    return np.concatenate(parts)

BINS = {
    'ph_topoetcone40': _var_bins(
        (-0.20, 0.30, 0.05),
        ( 0.30, 1.00, 0.10),
        ( 1.00, 2.00, 0.20),
        ( 2.00, 5.00, 0.50),
    ),
    'ph_topoetcone20': _var_bins(
        (-0.20, 0.50, 0.05),
        ( 0.50, 1.00, 0.10),
        ( 1.00, 2.00, 0.20),
        ( 2.00, 5.00, 0.50),
    ),
    'ph_ptcone20': _var_bins(
        ( 0.00, 0.30, 0.05),
        ( 0.30, 1.00, 0.10),
        ( 1.00, 2.00, 0.20),
        ( 2.00, 5.00, 0.50),
    ),
}

# pT bins in GeV: (label, lo_GeV, hi_GeV)  None = no upper bound
PT_BINS = [
    ('pt10to15',  10.,  15.),
    ('pt15to25',  15.,  25.),
    ('pt25to40',  25.,  40.),
    ('ptgt40',    40.,  None),
]

PT_LABELS = {
    'pt10to15': r'$10 < p_{\rm T}^{\gamma} < 15\ \mathrm{GeV}$',
    'pt15to25': r'$15 < p_{\rm T}^{\gamma} < 25\ \mathrm{GeV}$',
    'pt25to40': r'$25 < p_{\rm T}^{\gamma} < 40\ \mathrm{GeV}$',
    'ptgt40':   r'$p_{\rm T}^{\gamma} > 40\ \mathrm{GeV}$',
}

# Processes: tag used in accumulator, display label, filename substring(s) to match
PROCESSES = [
    ('Znunu',     '$Z(\nu\nu)$+jets', ['Znunu']),
    ('Wtaunu',    r'$W(\tau\nu)$+jets',        ['Wtaunu']),
    ('inclusive', 'Inclusive background',      None),   # None = accept all bkg MC
]
PROC_TAGS = [tag for tag, _, _ in PROCESSES]

ID_CRITERIA = ['Tight', 'LoosePrime4', 'Loose']

ID_STYLES = {
    'Tight':       {'color': '#1a237e', 'marker': 'o', 'label': 'Tight'},
    'LoosePrime4': {'color': '#b71c1c', 'marker': 's', 'label': 'LoosePrime4 (not Tight)'},
    'Loose':       {'color': '#1b5e20', 'marker': '^', 'label': 'Loose (not Tight)'},
}

NEEDED_BRANCHES = VARIABLES + [
    'ph_pt', 'ph_select_baseline', 'ph_select_tightID', 'ph_isEM', 'ph_truthJFP',
    'weight_total', 'weight_fjvt_effSF', 'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
    'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nPhotons_baseline', 'nPhotons_skims', 'nPhotons_baseline_noOR',
    'mTGammaMet', 'dPhiGammaMet', 'nTau20_veryloose', 'met_signif',
]

LP4_BIT   = LoosePrimeDefs['LoosePrime4']
LOOSE_BIT = LoosePrimeDefs['Loose']

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
    """Return list of process tags this file contributes to."""
    tags = []
    for tag, _, substrings in PROCESSES:
        if substrings is None:
            tags.append(tag)   # inclusive — always add
        elif any(s in fname for s in substrings):
            tags.append(tag)
    return tags

# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(run2):
    """Fill histograms for all (proc, pt_bin, id_crit, var)."""
    era_tag = 'Run2' if run2 else 'Run3'
    mc_tag  = 'mc20' if run2 else 'mc23'

    def _zeros(var):
        return np.zeros(len(BINS[var]) - 1)

    hists = {
        proc: {
            pt_tag: {id_c: {var: _zeros(var) for var in VARIABLES} for id_c in ID_CRITERIA}
            for pt_tag, _, _ in PT_BINS
        }
        for proc in PROC_TAGS
    }
    sumw2 = {
        proc: {
            pt_tag: {id_c: {var: _zeros(var) for var in VARIABLES} for id_c in ID_CRITERIA}
            for pt_tag, _, _ in PT_BINS
        }
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

        region_mask = get_region_masks(data)['Preselection']['0L']
        jfp_mask    = data['ph_truthJFP'] == 1

        baseline = data['ph_select_baseline'] == 1
        tight    = data['ph_select_tightID']  == 1
        lp4_shape   = (data['ph_isEM'] & LP4_BIT)   == 0
        loose_shape = (data['ph_isEM'] & LOOSE_BIT) == 0

        id_masks = {
            'Tight':       baseline &  tight,
            'LoosePrime4': baseline & ~tight & lp4_shape,
            'Loose':       baseline & ~tight & loose_shape,
        }

        pt_mev = data['ph_pt']
        pt_masks = {
            pt_tag: (pt_mev >= lo * 1e3) & (pt_mev < hi * 1e3 if hi is not None else True)
            for pt_tag, lo, hi in PT_BINS
        }

        for pt_tag, pt_mask in pt_masks.items():
            base = region_mask & jfp_mask & pt_mask
            for id_c, id_mask in id_masks.items():
                sel = base & id_mask
                if not sel.any():
                    continue
                w_sel = w[sel]
                for var in VARIABLES:
                    iso = data[var][sel]
                    if var == 'ph_topoetcone40':
                        iso = iso - 2450.   # subtract 2.45 GeV offset (branch in MeV)
                    raw    = iso / data['ph_pt'][sel]
                    values = np.minimum(raw, BINS[var][-1])
                    h,  _  = np.histogram(values, bins=BINS[var], weights=w_sel)
                    s2, _  = np.histogram(values, bins=BINS[var], weights=w_sel ** 2)
                    for tag in tags:
                        hists[tag][pt_tag][id_c][var] += h
                        sumw2[tag][pt_tag][id_c][var] += s2

    return hists, sumw2

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(hists, sumw2, proc_tag, proc_label, pt_tag, var,
              era_tag, lumi_label):
    bin_edges   = BINS[var]
    bin_widths  = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.06)
    gs       = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    tight_h   = hists[proc_tag][pt_tag]['Tight'][var]
    tight_s2  = sumw2[proc_tag][pt_tag]['Tight'][var]
    tight_den = tight_h  / bin_widths
    tight_err = np.sqrt(tight_s2) / bin_widths

    # --- Main panel ---
    ymax = 0.
    for id_c, style in ID_STYLES.items():
        h    = hists[proc_tag][pt_tag][id_c][var]
        s2   = sumw2[proc_tag][pt_tag][id_c][var]
        den  = h  / bin_widths
        derr = np.sqrt(s2) / bin_widths
        mask = den > 0
        if not mask.any():
            continue
        ax_main.errorbar(
            bin_centers[mask], den[mask], yerr=derr[mask],
            fmt=style['marker'], color=style['color'], label=style['label'],
            markersize=4, linewidth=0, elinewidth=1, capsize=2,
        )
        ymax = max(ymax, den[mask].max())

    ax_main.set_ylabel('Events / bin width', fontsize=16, labelpad=6, loc='top')
    ax_main.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.set_xlim(bin_edges[0], bin_edges[-1])
    ax_main.set_yscale('log')
    ax_main.set_ylim(bottom=1e-2)
    if ymax > 0:
        _, ytop = ax_main.get_ylim()
        log_range = np.log10(ytop) - np.log10(1e-2)
        ax_main.set_ylim(top=10 ** (np.log10(ytop) + 0.55 * log_range))

    # ATLAS label
    t_atlas = ax_main.text(0.05, 0.97, 'ATLAS',
                           transform=ax_main.transAxes, fontsize=14, va='top',
                           fontweight='bold', fontstyle='italic', fontfamily='Nimbus Sans')
    fig.canvas.draw()
    bb      = t_atlas.get_window_extent(renderer=fig.canvas.get_renderer())
    x1_axes = ax_main.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax_main.text(x1_axes + 0.01, 0.97, 'Internal',
                 transform=ax_main.transAxes, fontsize=14, va='top', fontfamily='Nimbus Sans')
    ax_main.text(0.05, 0.90, lumi_label,
                 transform=ax_main.transAxes, fontsize=11, va='top')
    ax_main.text(0.05, 0.84, f'Preselection / 0L  —  JFP',
                 transform=ax_main.transAxes, fontsize=11, va='top', fontfamily='monospace')
    ax_main.text(0.05, 0.78, proc_label,
                 transform=ax_main.transAxes, fontsize=11, va='top')
    ax_main.text(0.05, 0.72, PT_LABELS[pt_tag],
                 transform=ax_main.transAxes, fontsize=11, va='top')
    ax_main.legend(fontsize=12, framealpha=0.8, loc='upper right', bbox_to_anchor=(0.97, 0.97))
    ax_main.axvline(ISO_CUT[var], color='k', linestyle=':', linewidth=1.2)

    # --- Ratio panel ---
    ax_ratio.axhline(1., color='black', linewidth=0.8, linestyle='--')

    for id_c, style in ID_STYLES.items():
        if id_c == 'Tight':
            continue
        h    = hists[proc_tag][pt_tag][id_c][var]
        s2   = sumw2[proc_tag][pt_tag][id_c][var]
        den  = h  / bin_widths
        derr = np.sqrt(s2) / bin_widths

        valid      = (tight_den > 0) & (den > 0)
        safe_tight = np.where(valid, tight_den, 1.)
        safe_den   = np.where(valid, den,       1.)
        ratio      = np.where(valid, den / safe_tight, np.nan)
        ratio_err  = np.where(valid,
                               ratio * np.sqrt((derr       / safe_den  ) ** 2 +
                                               (tight_err  / safe_tight) ** 2),
                               np.nan)
        ax_ratio.errorbar(
            bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
            fmt=style['marker'], color=style['color'],
            markersize=4, linewidth=0, elinewidth=1, capsize=2,
        )

    ax_ratio.axvline(ISO_CUT[var], color='k', linestyle=':', linewidth=1.2)
    ax_ratio.set_xlabel(VAR_LABELS[var], fontsize=16, labelpad=6, loc='right')
    ax_ratio.set_ylabel('Ratio to Tight', fontsize=13, labelpad=6)
    ax_ratio.tick_params(axis='both', which='both', labelsize=13, direction='in')
    # Auto-scale upper limit: at least 6, expanded to fit visible points
    ratio_tops = []
    for id_c, style in ID_STYLES.items():
        if id_c == 'Tight':
            continue
        h   = hists[proc_tag][pt_tag][id_c][var]
        s2  = sumw2[proc_tag][pt_tag][id_c][var]
        den = h / bin_widths
        derr = np.sqrt(s2) / bin_widths
        valid = (tight_den > 0) & (den > 0)
        safe_tight = np.where(valid, tight_den, 1.)
        safe_den   = np.where(valid, den, 1.)
        r   = np.where(valid, den / safe_tight, np.nan)
        re  = np.where(valid, r * np.sqrt((derr / safe_den)**2 + (tight_err / safe_tight)**2), np.nan)
        ratio_tops.extend((r + re)[valid].tolist())
    ratio_max = max(ratio_tops) if ratio_tops else 0.
    ax_ratio.set_ylim(0, min(20., max(6., ratio_max * 1.2)))
    ax_ratio.yaxis.grid(True, color='lightgray', linewidth=0.7, zorder=0)
    ax_ratio.set_axisbelow(True)
    ax_ratio.axhline(1., color='black', linewidth=0.8, linestyle='--')

    fname = f'{era_tag}_{proc_tag}_{pt_tag}_{var}'
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for run2, era_tag, lumi_label in [
        (True,  'Run2', r'$\sqrt{s}=13\ \mathrm{TeV},\ 140\ \mathrm{fb}^{-1}$'),
        (False, 'Run3', r'$\sqrt{s}=13.6\ \mathrm{TeV},\ 160\ \mathrm{fb}^{-1}$'),
    ]:
        hists, sumw2 = accumulate(run2)

        n_plots = 0
        for proc_tag, proc_label, _ in PROCESSES:
            for pt_tag, _, _ in PT_BINS:
                for var in VARIABLES:
                    make_plot(hists, sumw2, proc_tag, proc_label, pt_tag, var,
                              era_tag, lumi_label)
                    n_plots += 1

        print(f'Saved {n_plots} plots for {era_tag} to {OUTPUT_DIR}/')
