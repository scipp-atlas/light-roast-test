#!/usr/bin/env python3
"""
Isolation distributions for JFP photons, overlaying Tight / LoosePrime4 / Loose
ID working points on a log-y scale with a ratio panel below.

Inclusive in pT, process, and campaign (Run 2 + Run 3 combined).
Runs over all available regions by default; use --regions to select a subset.

With --overlay-presel, a faded copy of the Preselection-0L shape (normalized to
unit area) is drawn over each target-region distribution for shape comparison.
The ratio panel then shows (target region) / (Preselection) per ID working point.
Without --overlay-presel (default), plots show weighted yields and the ratio panel
shows LP4/Tight and Loose/Tight within the target region.

Output: jfp_iso_region_plots/<region>_<var>[_vs_presel].{pdf,png}
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Nimbus Sans', 'FreeSans', 'DejaVu Sans']
import matplotlib.pyplot as plt
import numpy as np
import uproot

from abcd_utils import get_region_masks, LoosePrimeDefs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NTUPLE_DIR = '/data/mhance/SUSY/ntuples/v4.2'
OUTPUT_DIR = 'jfp_iso_region_plots'

VARIABLES = ['ph_topoetcone40', 'ph_topoetcone20', 'ph_ptcone20']

VAR_LABELS = {
    'ph_topoetcone40': (r'$(\mathrm{topo}E_{\rm T}^{\rm cone40}'
                        r' - 2.45\ \mathrm{GeV})\ /\ p_{\rm T}^{\gamma}$'),
    'ph_topoetcone20': r'$\mathrm{topo}E_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
    'ph_ptcone20':     r'$p_{\rm T}^{\rm cone20}\ /\ p_{\rm T}^{\gamma}$',
}

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
            edges = edges[1:]
        parts.append(edges)
    return np.concatenate(parts)

BINS = {
    'ph_topoetcone40': _var_bins(
        (-0.20, 0.30, 0.10),
        ( 0.30, 1.00, 0.20),
        ( 1.00, 2.00, 0.50),
        ( 2.00, 5.00, 1.00),
    ),
    'ph_topoetcone20': _var_bins(
        (-0.20, 0.50, 0.10),
        ( 0.50, 1.00, 0.20),
        ( 1.00, 2.00, 0.50),
        ( 2.00, 5.00, 1.00),
    ),
    'ph_ptcone20': _var_bins(
        ( 0.00, 0.30, 0.10),
        ( 0.30, 1.00, 0.20),
        ( 1.00, 2.00, 0.50),
        ( 2.00, 5.00, 1.00),
    ),
}

# All available regions: (region_type, region_name, display_label, file_key)
ALL_REGIONS = [
    ('Preselection', '0L',              'Preselection 0L',      'Presel-0L'),
    ('VR',           '0L-mT-mid',       'VR 0L-mT-mid',         'VR-0L-mT-mid'),
    ('SR',           '0L-mT-low',       'SR 0L-mT-low',         'SR-0L-mT-low'),
    ('SR',           '0L-mT-mid',       'SR 0L-mT-mid',         'SR-0L-mT-mid'),
    ('SR',           '0L-mT-hgh',       'SR 0L-mT-hgh',         'SR-0L-mT-hgh'),
    ('SR',           '0L-mT-low-loose', 'SR 0L-mT-low-loose',   'SR-0L-mT-low-loose'),
    ('SR',           '0L-mT-mid-loose', 'SR 0L-mT-mid-loose',   'SR-0L-mT-mid-loose'),
    ('SR',           '0L-mT-hgh-loose', 'SR 0L-mT-hgh-loose',   'SR-0L-mT-hgh-loose'),
]

PRESEL_KEY      = 'Presel-0L'
REGION_BY_KEY   = {r[3]: r for r in ALL_REGIONS}
ALL_REGION_KEYS = [r[3] for r in ALL_REGIONS]

ID_CRITERIA = ['Tight', 'LoosePrime4', 'Loose']

ID_STYLES = {
    'Tight':       {'color': '#1a237e', 'marker': 'o', 'label': 'Tight'},
    'LoosePrime4': {'color': '#b71c1c', 'marker': 's', 'label': 'LoosePrime4 (not Tight)'},
    'Loose':       {'color': '#1b5e20', 'marker': '^', 'label': 'Loose (not Tight)'},
}

LP4_BIT   = LoosePrimeDefs['LoosePrime4']
LOOSE_BIT = LoosePrimeDefs['Loose']

NEEDED_BRANCHES = VARIABLES + [
    'ph_pt', 'ph_select_baseline', 'ph_select_tightID', 'ph_isEM',
    'ph_truthJFP', 'ph_truthother',
    'weight_total', 'weight_fjvt_effSF', 'weight_ftag_effSF_GN2v01_Continuous',
    'weight_jvt_effSF',
    'met_met', 'jet_cleanTightBad_prod', 'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nPhotons_baseline', 'nPhotons_skims',
    'nPhotons_baseline_noOR', 'mTGammaMet', 'dPhiGammaMet', 'nTau20_veryloose',
    'met_signif',
]

_SKIP = ('gammajet', 'jetjet', '_jj_', 'N2_', 'signal', 'data_')

# Process substrings used for --processes filtering; None means accept all bkg MC
PROCESS_SUBSTRINGS = {
    'Wtaunu': ['Wtaunu'],
    'Znunu':  ['Znunu'],
}

LUMI_LABEL = (r'$\sqrt{s}=13\text{--}13.6\ \mathrm{TeV},$'
              r'$\ \mathrm{Run\ 2+3\ combined}$')

def is_background_mc(fname, processes=None):
    if any(s in fname for s in _SKIP):
        return False
    if 'mc20' not in fname and 'mc23' not in fname:
        return False
    if processes is not None:
        substrings = [s for p in processes for s in PROCESS_SUBSTRINGS[p]]
        if not any(s in fname for s in substrings):
            return False
    return True

# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def accumulate(region_keys, truth, processes=None):
    """
    Fill histograms for the selected truth category/categories for all requested
    regions × ID × variable, combining all background MC files from both Run 2
    and Run 3.

    truth : one of 'jfp', 'other', 'jfp+other'

    Preselection-0L is always accumulated (needed for --overlay-presel).

    Returns:
        hists[region_key][id_crit][var]  -> summed-weight array
        sumw2[region_key][id_crit][var]  -> sum-of-weights-squared array
    """
    keys_to_fill = set(region_keys) | {PRESEL_KEY}

    def _zeros(var):
        return np.zeros(len(BINS[var]) - 1)

    hists = {
        rk: {id_c: {var: _zeros(var) for var in VARIABLES} for id_c in ID_CRITERIA}
        for rk in keys_to_fill
    }
    sumw2 = {
        rk: {id_c: {var: _zeros(var) for var in VARIABLES} for id_c in ID_CRITERIA}
        for rk in keys_to_fill
    }

    all_files = sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root')))
    bkg_files = [f for f in all_files if is_background_mc(os.path.basename(f), processes)]
    proc_desc = ', '.join(processes) if processes else 'all processes'
    print(f'\nAccumulating {len(bkg_files)} background MC files ({proc_desc}, Run 2 + Run 3 combined)')

    for i, fp in enumerate(bkg_files, 1):
        fname = os.path.basename(fp)
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

        region_masks_all = get_region_masks(data)

        jfp_mask   = data['ph_truthJFP']   == 1
        other_mask = data['ph_truthother'] == 1
        if truth == 'jfp':
            truth_mask = jfp_mask
        elif truth == 'other':
            truth_mask = other_mask
        else:  # jfp+other
            truth_mask = jfp_mask | other_mask

        baseline    = data['ph_select_baseline'] == 1
        tight       = data['ph_select_tightID']  == 1
        lp4_shape   = (data['ph_isEM'] & LP4_BIT)   == 0
        loose_shape = (data['ph_isEM'] & LOOSE_BIT) == 0

        id_masks = {
            'Tight':       baseline &  tight,
            'LoosePrime4': baseline & ~tight & lp4_shape,
            'Loose':       baseline & ~tight & loose_shape,
        }

        for rk in keys_to_fill:
            rtype, rname, _, _ = REGION_BY_KEY[rk]
            region_mask = region_masks_all[rtype][rname]
            base = region_mask & truth_mask

            for id_c, id_mask in id_masks.items():
                sel = base & id_mask
                if not sel.any():
                    continue
                w_sel = w[sel]
                for var in VARIABLES:
                    iso = data[var][sel].copy()
                    if var == 'ph_topoetcone40':
                        iso = iso - 2450.   # subtract 2.45 GeV offset (branch in MeV)
                    raw    = iso / data['ph_pt'][sel]
                    values = np.minimum(raw, BINS[var][-1])
                    h,  _  = np.histogram(values, bins=BINS[var], weights=w_sel)
                    s2, _  = np.histogram(values, bins=BINS[var], weights=w_sel ** 2)
                    hists[rk][id_c][var] += h
                    sumw2[rk][id_c][var] += s2

    return hists, sumw2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normdensity(h, s2, bin_widths):
    """Return (density, density_err) normalized to unit area."""
    total = h.sum()
    if total == 0:
        return np.zeros_like(h), np.zeros_like(h)
    den = h  / total / bin_widths
    err = np.sqrt(s2) / total / bin_widths
    return den, err

def _rawdensity(h, s2, bin_widths):
    """Return (density, density_err) as weighted yield / bin width."""
    return h / bin_widths, np.sqrt(s2) / bin_widths

def _atlas_label(ax, fig, lumi, region_label, extra_lines=()):
    """Draw ATLAS Internal + lumi + region label in the top-left of ax."""
    t = ax.text(0.05, 0.97, 'ATLAS',
                transform=ax.transAxes, fontsize=14, va='top',
                fontweight='bold', fontstyle='italic', fontfamily='sans-serif')
    fig.canvas.draw()
    bb      = t.get_window_extent(renderer=fig.canvas.get_renderer())
    x1      = ax.transAxes.inverted().transform((bb.x1, bb.y0))[0]
    ax.text(x1 + 0.01, 0.97, 'Internal',
            transform=ax.transAxes, fontsize=14, va='top', fontfamily='sans-serif')
    y = 0.90
    ax.text(0.05, y, lumi, transform=ax.transAxes, fontsize=10, va='top')
    y -= 0.07
    ax.text(0.05, y, region_label,
            transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace')
    for line in extra_lines:
        y -= 0.06
        ax.text(0.05, y, line, transform=ax.transAxes, fontsize=9, va='top', color='gray')

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_TRUTH_LABEL = {
    'jfp':      'JFP photons',
    'other':    'Other photons',
    'jfp+other': 'JFP + Other photons',
}


def make_plot(hists, sumw2, region_key, var, overlay_presel, truth):
    _, _, region_label, _ = REGION_BY_KEY[region_key]
    do_overlay = overlay_presel and region_key != PRESEL_KEY

    bin_edges   = BINS[var]
    bin_widths  = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.06)
    gs       = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # Pre-compute Tight reference for ratio panel
    tight_h  = hists[region_key]['Tight'][var]
    tight_s2 = sumw2[region_key]['Tight'][var]
    if do_overlay:
        tight_den, tight_err = _normdensity(tight_h, tight_s2, bin_widths)
    else:
        tight_den, tight_err = _rawdensity(tight_h, tight_s2, bin_widths)

    ymax = 0.

    # --- Main panel: target-region distributions ---
    for id_c, style in ID_STYLES.items():
        h  = hists[region_key][id_c][var]
        s2 = sumw2[region_key][id_c][var]

        if do_overlay:
            den, err = _normdensity(h, s2, bin_widths)
        else:
            den, err = _rawdensity(h, s2, bin_widths)

        mask = den > 0
        if mask.any():
            ax_main.errorbar(
                bin_centers[mask], den[mask], yerr=err[mask],
                fmt=style['marker'], color=style['color'], label=style['label'],
                markersize=4, linewidth=0, elinewidth=1, capsize=2, zorder=3,
            )
            ymax = max(ymax, den[mask].max())

    # --- Main panel: faded presel overlay (shape-normalized) ---
    if do_overlay:
        for id_c, style in ID_STYLES.items():
            ph  = hists[PRESEL_KEY][id_c][var]
            ps2 = sumw2[PRESEL_KEY][id_c][var]
            pden, perr = _normdensity(ph, ps2, bin_widths)
            pmask = pden > 0
            if pmask.any():
                ax_main.errorbar(
                    bin_centers[pmask], pden[pmask], yerr=perr[pmask],
                    fmt=style['marker'], color=style['color'],
                    alpha=0.30, markersize=3, linewidth=0, elinewidth=0.7, capsize=1.5,
                    zorder=2,
                )
                ymax = max(ymax, pden[pmask].max())

    # --- Main panel axis formatting ---
    ylabel = 'Normalized / bin width' if do_overlay else 'Events / bin width'
    ax_main.set_ylabel(ylabel, fontsize=16, labelpad=6, loc='top')
    ax_main.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.set_xlim(bin_edges[0], bin_edges[-1])
    ax_main.set_yscale('log')
    ax_main.set_ylim(bottom=1e-4)
    if ymax > 0:
        _, ytop = ax_main.get_ylim()
        log_range = np.log10(ytop) - np.log10(1e-4)
        ax_main.set_ylim(top=10 ** (np.log10(ytop) + 0.60 * log_range))

    extra_lines = []
    extra_lines.append(f'{_TRUTH_LABEL[truth]}, inclusive in pT / process')
    if do_overlay:
        extra_lines.append('Solid: this region   Faded: Preselection-0L')

    _atlas_label(ax_main, fig, LUMI_LABEL, region_label, extra_lines)
    ax_main.legend(fontsize=12, framealpha=0.8, loc='upper right',
                   bbox_to_anchor=(0.97, 0.97))
    ax_main.axvline(ISO_CUT[var], color='k', linestyle=':', linewidth=1.2)

    # --- Ratio panel ---
    ax_ratio.axhline(1., color='black', linewidth=0.8, linestyle='--')

    ratio_tops = []

    if do_overlay:
        # Ratio = (normalized target region) / (normalized presel) for each ID WP
        ratio_label = 'Region / Presel'
        for id_c, style in ID_STYLES.items():
            h  = hists[region_key][id_c][var]
            s2 = sumw2[region_key][id_c][var]
            ph  = hists[PRESEL_KEY][id_c][var]
            ps2 = sumw2[PRESEL_KEY][id_c][var]

            den,  err  = _normdensity(h,  s2,  bin_widths)
            pden, perr = _normdensity(ph, ps2, bin_widths)

            valid      = (den > 0) & (pden > 0)
            safe_den   = np.where(valid, den,  1.)
            safe_pden  = np.where(valid, pden, 1.)
            ratio      = np.where(valid, den / safe_pden, np.nan)
            ratio_err  = np.where(valid,
                                   ratio * np.sqrt((err  / safe_den ) ** 2 +
                                                   (perr / safe_pden) ** 2),
                                   np.nan)
            ax_ratio.errorbar(
                bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
                fmt=style['marker'], color=style['color'],
                markersize=4, linewidth=0, elinewidth=1, capsize=2,
            )
            good = valid & np.isfinite(ratio)
            if good.any():
                ratio_tops.extend((ratio + ratio_err)[good].tolist())

    else:
        # Standard ratio: LP4/Tight and Loose/Tight within this region
        ratio_label = 'Ratio to Tight'
        for id_c, style in ID_STYLES.items():
            if id_c == 'Tight':
                continue
            h  = hists[region_key][id_c][var]
            s2 = sumw2[region_key][id_c][var]
            den, derr = _rawdensity(h, s2, bin_widths)

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
            good = valid & np.isfinite(ratio)
            if good.any():
                ratio_tops.extend((ratio + ratio_err)[good].tolist())

    ax_ratio.axvline(ISO_CUT[var], color='k', linestyle=':', linewidth=1.2)
    ax_ratio.set_xlabel(VAR_LABELS[var], fontsize=16, labelpad=6, loc='right')
    ax_ratio.set_ylabel(ratio_label, fontsize=12, labelpad=6)
    ax_ratio.tick_params(axis='both', which='both', labelsize=13, direction='in')
    ax_ratio.yaxis.grid(True, color='lightgray', linewidth=0.7, zorder=0)
    ax_ratio.set_axisbelow(True)
    ax_ratio.axhline(1., color='black', linewidth=0.8, linestyle='--')

    ratio_max  = max(ratio_tops) if ratio_tops else 0.
    ratio_ymin = 3. if do_overlay else 6.
    ax_ratio.set_ylim(0, min(20., max(ratio_ymin, ratio_max * 1.2)))

    suffix = '_vs_presel' if do_overlay else ''
    fname  = f'{region_key}_{var}{suffix}'
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'{fname}.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# HTML index
# ---------------------------------------------------------------------------

_CSS = """
  body { font-family: sans-serif; background: #f4f4f4; margin: 0; padding: 16px; }
  h1 { margin-bottom: 4px; }
  p.desc { color: #555; margin: 0 0 16px 0; }
  h2 { background: #2c3e50; color: white; padding: 8px 14px;
       border-radius: 4px; margin: 32px 0 8px 0; }
  .table { display: grid; gap: 8px; margin-bottom: 16px; }
  .col-header { text-align: center; font-weight: bold; font-size: 12px;
                color: #333; background: #dce4f0; border-radius: 4px;
                padding: 4px 2px; align-self: center; }
  .row-label { display: flex; align-items: center; justify-content: center;
               text-align: center; font-size: 11px; font-weight: bold;
               color: #555; background: #eee; border-radius: 4px;
               padding: 4px; line-height: 1.3; }
  .row-label.shape { background: #fde8cc; }
  .corner { background: transparent; }
  .cell { background: white; border-radius: 6px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.15); padding: 6px; }
  .missing { color: #999; font-size: 11px; font-style: italic;
             padding: 8px 4px; text-align: center; }
  img { width: 100%; height: auto; display: block;
        border-radius: 3px; cursor: zoom-in; }
  #lightbox { display: none; position: fixed; inset: 0;
              background: rgba(0,0,0,0.85); z-index: 1000;
              justify-content: center; align-items: center; }
  #lightbox.active { display: flex; }
  #lightbox img { max-width: 90vw; max-height: 90vh; width: auto;
                  border-radius: 6px; cursor: default;
                  box-shadow: 0 4px 32px rgba(0,0,0,0.6); }
  #lightbox-close { position: fixed; top: 18px; right: 28px;
                    font-size: 36px; color: white; cursor: pointer;
                    line-height: 1; user-select: none; }
"""

_JS = """
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('active');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}
document.getElementById('lightbox').addEventListener('click', function(e) {
  if (e.target === this) closeLightbox();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeLightbox();
});
"""

_VAR_HTML_LABELS = {
    'ph_topoetcone40': ('(topoE<sub>T</sub><sup>cone40</sup> &minus; 2.45&thinsp;GeV)'
                        ' / p<sub>T</sub><sup>&gamma;</sup>'),
    'ph_topoetcone20': 'topoE<sub>T</sub><sup>cone20</sup> / p<sub>T</sub><sup>&gamma;</sup>',
    'ph_ptcone20':     'p<sub>T</sub><sup>cone20</sup> / p<sub>T</sub><sup>&gamma;</sup>',
}

# Rows shown in the index, in order.  Suffix '' = yield mode; '_vs_presel' = shape mode.
_MODES = [
    ('',           'Weighted yields',                     ''),
    ('_vs_presel', 'Shape comparison (vs Preselection)',  'shape'),
]


def write_index(region_keys, out_dir):
    import base64

    def _img_tag(fname):
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            return f'<div class="cell missing">{fname}</div>'
        with open(fpath, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return (f'<div class="cell">'
                f'<img src="data:image/png;base64,{b64}" '
                f'onclick="openLightbox(this.src)" title="{fname}">'
                f'</div>')

    lines = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>JFP Isolation by Region</title>',
        f'<style>{_CSS}</style>',
        '</head>',
        '<body>',
        '<h1>JFP Isolation Distributions by Region</h1>',
        '<p class="desc">Tight / LoosePrime4 / Loose overlaid for JFP photons, '
        'inclusive in p<sub>T</sub>, process, and campaign (Run&thinsp;2+3 combined). '
        'Click any plot to enlarge.</p>',
        '<div id="lightbox">',
        '  <span id="lightbox-close" onclick="closeLightbox()">&#x2715;</span>',
        '  <img id="lightbox-img" src="" alt="">',
        '</div>',
        f'<script>{_JS}</script>',
    ]

    for var in VARIABLES:
        var_label = _VAR_HTML_LABELS[var]
        lines.append(f'<h2>{var_label}</h2>')

        # Check which modes have at least one existing file for this variable
        active_modes = []
        for suffix, mode_label, tint in _MODES:
            if any(os.path.exists(os.path.join(out_dir, f'{rk}_{var}{suffix}.png'))
                   for rk in region_keys):
                active_modes.append((suffix, mode_label, tint))

        if not active_modes:
            lines.append('<p style="color:#999;font-style:italic">No plots found.</p>')
            continue

        # Columns = modes; rows = regions
        n_mode_cols = len(active_modes)
        col_tpl = f'140px repeat({n_mode_cols}, 1fr)'
        lines.append(f'<div class="table" style="grid-template-columns:{col_tpl};">')

        # Header row: empty corner + one column header per mode
        lines.append('<div class="corner"></div>')
        for suffix, mode_label, tint in active_modes:
            lines.append(f'<div class="col-header">{mode_label}</div>')

        # One row per region
        for rk in region_keys:
            _, _, reg_label, _ = REGION_BY_KEY[rk]
            lines.append(f'<div class="row-label">{reg_label}</div>')
            for suffix, mode_label, tint in active_modes:
                lines.append(_img_tag(f'{rk}_{var}{suffix}.png'))

        lines.append('</div>')  # .table

    lines += ['</body>', '</html>']

    out_path = os.path.join(out_dir, 'index.html')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'Written {out_path} ({os.path.getsize(out_path) // 1024} KB)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Isolation distributions for JFP photons overlaying Tight/LP4/Loose, '
            'inclusive in pT/process/campaign (Run 2 + Run 3 combined).'
        )
    )
    parser.add_argument(
        '--regions', nargs='+', default=ALL_REGION_KEYS,
        metavar='REGION',
        help=(
            'Regions to plot. '
            f'Valid choices: {ALL_REGION_KEYS}. '
            'Default: all regions.'
        ),
    )
    parser.add_argument(
        '--index-only', action='store_true',
        help='Regenerate index.html from existing PNG files without re-running plots.',
    )
    parser.add_argument(
        '--processes', nargs='+', choices=list(PROCESS_SUBSTRINGS), default=None,
        metavar='PROC',
        help=(
            'Restrict to specific MC processes. '
            f'Choices: {list(PROCESS_SUBSTRINGS)}. '
            'Default: all background MC samples.'
        ),
    )
    parser.add_argument(
        '--truth', choices=['jfp', 'other', 'jfp+other'], default='jfp+other',
        help='Truth category to include: jfp, other, or jfp+other (default: jfp+other).',
    )
    parser.add_argument(
        '--overlay-presel', action='store_true',
        help=(
            'Overlay the Preselection-0L shape (normalized to unit area, faded) '
            'for shape comparison. Ratio panel shows (target region) / (Preselection). '
            'Default (off): show weighted yields; ratio panel shows LP4/Tight and '
            'Loose/Tight within the target region.'
        ),
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    invalid = [r for r in args.regions if r not in REGION_BY_KEY]
    if invalid:
        raise SystemExit(
            f'Unknown region(s): {invalid}\n'
            f'Valid choices: {ALL_REGION_KEYS}'
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.index_only:
        hists, sumw2 = accumulate(args.regions, args.truth, args.processes)

        n_plots = 0
        for rk in args.regions:
            for var in VARIABLES:
                make_plot(hists, sumw2, rk, var, args.overlay_presel, args.truth)
                n_plots += 1

        print(f'\nSaved {n_plots} plots to {OUTPUT_DIR}/')

    write_index(args.regions, OUTPUT_DIR)
