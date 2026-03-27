#!/usr/bin/env python3
"""Generate index.html for jfp_composition_plots with all PNGs embedded as base64."""

import base64
import os

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Regions become columns
REGIONS = [
    ('Preselection_0L',    'Preselection 0L'),
    ('VR_0L_mT_mid',       'VR 0L-mT-mid'),
    ('SR_0L_mT_low_loose', 'SR 0L-mT-low-loose'),
    ('SR_0L_mT_mid_loose', 'SR 0L-mT-mid-loose'),
    ('SR_0L_mT_hgh_loose', 'SR 0L-mT-hgh-loose'),
]

VARIABLES = [
    ('ph_pt',           'p<sub>T</sub><sup>&gamma;</sup>'),
    ('ph_eta',          '&eta;<sup>&gamma;</sup>'),
    ('ph_topoetcone40', 'topoE<sub>T</sub><sup>cone40</sup> / p<sub>T</sub><sup>&gamma;</sup>'),
    ('ph_topoetcone20', 'topoE<sub>T</sub><sup>cone20</sup> / p<sub>T</sub><sup>&gamma;</sup>'),
    ('ph_ptcone20',     'p<sub>T</sub><sup>cone20</sup> / p<sub>T</sub><sup>&gamma;</sup>'),
    ('ph_truthOrigin',  'photon truthOrigin'),
    ('ph_truthType',    'photon truthType'),
    ('dPhiGammaMet',    '&Delta;&phi;(&gamma;, E<sub>T</sub><sup>miss</sup>)'),
    ('dPhiGammaJ1',     '&Delta;&phi;(&gamma;, j<sub>1</sub>)'),
    ('mTGammaMet',      'm<sub>T</sub>(&gamma;, E<sub>T</sub><sup>miss</sup>)'),
]

# Rows: (ID cat, era) combinations
ID_CATS = [('Tight', 'Tight ID'), ('LP4', 'LoosePrime4')]
ERAS    = [('Run2',  'Run 2'),    ('Run3', 'Run 3')]
ROWS    = [(id_tag, id_label, era_tag, era_label)
           for id_tag, id_label in ID_CATS
           for era_tag, era_label in ERAS]


def img_tag(fname):
    fpath = os.path.join(PLOT_DIR, fname)
    if not os.path.exists(fpath):
        return f'<div class="cell missing">{fname}</div>'
    with open(fpath, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        f'<div class="cell">'
        f'<img src="data:image/png;base64,{b64}" '
        f'onclick="openLightbox(this.src)" title="{fname}">'
        f'</div>'
    )


CSS = """
  body { font-family: sans-serif; background: #f4f4f4; margin: 0; padding: 16px; }
  h1 { margin-bottom: 4px; }
  h2 { background: #2c3e50; color: white; padding: 8px 14px;
       border-radius: 4px; margin: 32px 0 8px 0; }
  /* 6-column grid: narrow row-label column + 5 equal region columns */
  .table { display: grid;
           grid-template-columns: 90px repeat(5, 1fr);
           gap: 8px; margin-bottom: 16px; }
  .col-header { text-align: center; font-weight: bold;
                font-size: 12px; color: #333;
                background: #dce4f0; border-radius: 4px;
                padding: 4px 2px; align-self: center; }
  .row-label { display: flex; align-items: center; justify-content: center;
               text-align: center; font-size: 11px; font-weight: bold;
               color: #555; background: #eee; border-radius: 4px;
               padding: 4px; line-height: 1.3; }
  .row-label.lp4 { background: #fde8cc; }   /* warm tint for LP4 rows */
  .row-label.tight { background: #d4e8fd; } /* cool tint for Tight rows */
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

JS = """
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

lines = [
    '<!DOCTYPE html>',
    '<html lang="en">',
    '<head>',
    '<meta charset="UTF-8">',
    '<title>JFP/Other Process Composition Plots</title>',
    f'<style>{CSS}</style>',
    '</head>',
    '<body>',
    '<h1>JFP / Other Process Composition Plots</h1>',
    '<p>Stacked histograms of JFP and Other photon backgrounds split by process '
    '(Znunu, Wtaunu, Wmunu, Wenu). '
    'Columns = regions; rows = ID working point &times; run era. '
    'Click any plot to enlarge.</p>',
    '<div id="lightbox">',
    '  <span id="lightbox-close" onclick="closeLightbox()">&#x2715;</span>',
    '  <img id="lightbox-img" src="" alt="">',
    '</div>',
    f'<script>{JS}</script>',
]

for var_key, var_label in VARIABLES:
    lines.append(f'<h2>{var_label}</h2>')
    lines.append('<div class="table">')

    # Header row: empty corner + one column header per region
    lines.append('<div class="corner"></div>')
    for _, reg_label in REGIONS:
        lines.append(f'<div class="col-header">{reg_label}</div>')

    # Data rows: one per (ID cat, era)
    for id_tag, id_label, era_tag, era_label in ROWS:
        tint = 'tight' if id_tag == 'Tight' else 'lp4'
        lines.append(f'<div class="row-label {tint}">{id_label}<br>{era_label}</div>')
        for reg_tag, _ in REGIONS:
            lines.append(img_tag(f'{reg_tag}_{var_key}_{id_tag}_{era_tag}.png'))

    lines.append('</div>')  # .table

lines += ['</body>', '</html>']

out = os.path.join(PLOT_DIR, 'index.html')
with open(out, 'w') as f:
    f.write('\n'.join(lines))

print(f"Written {out} ({os.path.getsize(out) // 1024} KB)")
