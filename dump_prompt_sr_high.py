#!/usr/bin/env python3
"""
Dump kinematics of prompt Tight photon events in SR/0L-mT-hgh-loose
from Run 3 W(taunu)+jets MC, to understand why prompt photons appear there.
"""

import glob
import os
import numpy as np
import uproot

from abcd_utils import get_region_masks

NTUPLE_DIR = '/data/mhance/SUSY/ntuples/v4.2'

BRANCHES = [
    'ph_pt', 'ph_eta', 'ph_phi', 'ph_truthpt',
    'ph_select_baseline', 'ph_select_tightID',
    'ph_truthprompt', 'ph_truthEFP', 'ph_truthJFP', 'ph_truthother',
    'met_met', 'met_signif', 'mTGammaMet', 'dPhiGammaMet',
    'j1_pt', 'mindPhiJetMet', 'nBTagJets',
    'nElectrons', 'nMuons', 'nTau20_veryloose', 'nPhotons_baseline',
    'weight_total', 'weight_fjvt_effSF',
    'weight_ftag_effSF_GN2v01_Continuous', 'weight_jvt_effSF',
    'jet_cleanTightBad_prod',
    'nPhotons_skims', 'nPhotons_baseline_noOR',
]

COLUMNS = [
    ('file',       's',  18),
    ('weight',     'g',  12),
    ('ph_pt',      '.1f', 8),
    ('ph_eta',     '.2f', 6),
    ('ph_phi',     '.2f', 6),
    ('truthpt',    '.1f', 8),
    ('truth',      's',   8),
    ('MET',        '.1f', 7),
    ('METsig',     '.1f', 7),
    ('mT',         '.1f', 7),
    ('dPhiGMET',   '.2f', 9),
    ('j1_pt',      '.1f', 7),
    ('minDPhiJM',  '.2f', 10),
    ('nBTag',      'd',   6),
    ('nEl',        'd',   4),
    ('nMu',        'd',   4),
    ('nTau',       'd',   5),
    ('nPh',        'd',   4),
]


def fmt(val, spec, width):
    s = format(val, spec)
    return s[:width].ljust(width) if spec == 's' else s[:width].rjust(width)


def print_header():
    parts = [fmt(name, 's', w) for name, _, w in COLUMNS]
    line  = '  '.join(parts)
    print(line)
    print('-' * len(line))


def print_row(row):
    parts = [fmt(row[name], spec, w) for name, spec, w in COLUMNS]
    print('  '.join(parts))


# ---------------------------------------------------------------------------

all_files = sorted(glob.glob(os.path.join(NTUPLE_DIR, '*.root')))
wtaunu_files = [f for f in all_files
                if 'mc23' in os.path.basename(f)
                and 'Wtaunu' in os.path.basename(f)]

print(f'Found {len(wtaunu_files)} Run 3 Wtaunu files\n')

rows = []

for fp in wtaunu_files:
    fname = os.path.basename(fp)
    short = fname.replace('.root', '')[-18:]
    try:
        with uproot.open(fp) as uf:
            if 'picontuple' not in uf:
                continue
            data = uf['picontuple'].arrays(BRANCHES, library='np')
    except Exception as exc:
        print(f'ERROR {fname}: {exc}')
        continue

    region_masks = get_region_masks(data)
    try:
        sr_high = region_masks['SR']['0L-mT-hgh-loose']
    except KeyError:
        continue

    tight  = (data['ph_select_baseline'] == 1) & (data['ph_select_tightID'] == 1)
    prompt = data['ph_truthprompt'] == 1
    sel    = sr_high & tight & prompt

    if not sel.any():
        continue

    w = (data['weight_total'] *
         data['weight_fjvt_effSF'] *
         data['weight_ftag_effSF_GN2v01_Continuous'] *
         data['weight_jvt_effSF'])

    for i in np.where(sel)[0]:
        truth_str = ''.join(
            lbl for branch, lbl in [
                ('ph_truthprompt', 'P'), ('ph_truthEFP', 'E'),
                ('ph_truthJFP', 'J'),    ('ph_truthother', 'O'),
            ] if data[branch][i]
        ) or '-'

        rows.append({
            'file':      short,
            'weight':    w[i],
            'ph_pt':     data['ph_pt'][i]      / 1e3,
            'ph_eta':    data['ph_eta'][i],
            'ph_phi':    data['ph_phi'][i],
            'truthpt':   data['ph_truthpt'][i] / 1e3,
            'truth':     truth_str,
            'MET':       data['met_met'][i]    / 1e3,
            'METsig':    data['met_signif'][i],
            'mT':        data['mTGammaMet'][i] / 1e3,
            'dPhiGMET':  data['dPhiGammaMet'][i],
            'j1_pt':     data['j1_pt'][i]      / 1e3,
            'minDPhiJM': data['mindPhiJetMet'][i],
            'nBTag':     int(data['nBTagJets'][i]),
            'nEl':       int(data['nElectrons'][i]),
            'nMu':       int(data['nMuons'][i]),
            'nTau':      int(data['nTau20_veryloose'][i]),
            'nPh':       int(data['nPhotons_baseline'][i]),
        })

if rows:
    print_header()
    for row in rows:
        print_row(row)
    print()

print(f'Total: {len(rows)} prompt Tight photon event(s) in SR-high Run3 Wtaunu')
