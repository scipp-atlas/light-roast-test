import uproot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import json
import glob
import re

_RUN2_YEARS = {2015, 2016, 2017, 2018}
_RUN3_YEARS = {2022, 2023, 2024}

def _parse_data_year(fname):
    """Extract year from a data filename, supporting both data_2017 and data_17 forms."""
    m = re.search(r'_data_(\d{2,4})_', fname)
    if not m:
        return None
    y = int(m.group(1))
    return y + 2000 if y < 100 else y

def _is_run2(fp):
    fname = os.path.basename(fp)
    if "mc20" in fname:
        return True
    year = _parse_data_year(fname)
    return year in _RUN2_YEARS if year is not None else False

def _is_run3(fp):
    fname = os.path.basename(fp)
    if "mc23" in fname:
        return True
    year = _parse_data_year(fname)
    return year in _RUN3_YEARS if year is not None else False

# https://docs.google.com/document/d/1sF0uQq8MA08Dbmd2euJ9BSo1nGIrc78XAOjKQy_mimA/edit?tab=t.rq6d75o3ywqx
LoosePrimeDefs = {
    "LoosePrimeRun1": 0x45fc01,
    "LoosePrime2":    0x27fc00,
    "LoosePrime3":    0x25fc00,
    "LoosePrime4":    0x05fc00,
    "LoosePrime4a":   0x21fc00,
    "LoosePrime5":    0x01fc00,
    "Loose":          0x000000,
}

# Default signal samples used in printregion; override by passing sigsamples= explicitly.
DEFAULT_SIGSAMPLES = [
    "N2_200_N1_185_WB",
    "N2_200_N1_190_WB",
    "N2_200_N1_195_WB",
    "N2_200_N1_197_WB",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def load_json_file(file_path):
    """
    Loads JSON data from a file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: A Python dictionary or list representing the JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def getemptyresults():
    results = {}
    for b in ['TT', 'TL', 'LT', 'LL']:
        results[b] = {'mc': {'nevents': 0,
                             'sumweights': 0,
                             'staterr': 0},
                      'real': {'nevents': 0,
                               'sumweights': 0,
                               'staterr': 0},
                      'jfp': {'nevents': 0,
                              'sumweights': 0,
                              'staterr': 0},
                      'efp': {'nevents': 0,
                              'sumweights': 0,
                              'staterr': 0},
                      'other': {'nevents': 0,
                                'sumweights': 0,
                                'staterr': 0},
                      'unclassified': {'nevents': 0,
                                       'sumweights': 0,
                                       'staterr': 0},
                      'data': {'nevents': 0, 'sumweights': 0.0, 'staterr': 0.0}
                     }
    return results


def get_region_masks(data):
    """Return event-level selection masks for each analysis region."""
    PS_0L = (
        (data['met_met']                >  200*1000. ) &
        (data['jet_cleanTightBad_prod'] == 1         ) &
        (data['j1_pt']                  >  150*1000. ) &
        (data['ph_pt']                  >   10*1000. ) &
        (data['mindPhiJetMet']          >  0.4       ) &
        (data['nBTagJets']              == 0         ) &
        (data['nElectrons']             == 0         ) &
        (data['nMuons']                 == 0         ) &
        (data['nPhotons_baseline']      == 1         ) &
        (data['nPhotons_skims']         == 1         ) &
        (data['nPhotons_baseline_noOR'] == 1         ) &
        (data['nTau20_veryloose']       == 0         )
    )

    SR_0L_mT_low = PS_0L & (
        (data['mTGammaMet']       <   50*1000.) &
        (data['met_signif']       >   25      )
    )

    SR_0L_mT_mid = PS_0L & (
        (data['mTGammaMet']       >   50*1000.) &
        (data['mTGammaMet']       <  100*1000.) &
        (data['met_signif']       >   21      ) &
        (data['dPhiGammaMet']     <  0.7      )
    )

    SR_0L_mT_hgh = PS_0L & (
        (data['mTGammaMet']       >  100*1000.) &
        (data['met_signif']       >   19      ) &
        (data['dPhiGammaMet']     <  1.0      )
    )

    SR_0L_mT_low_loose = PS_0L & (
        (data['mTGammaMet']       <   50*1000.) &
        (data['met_signif']       >   18      )
    )

    SR_0L_mT_mid_loose = PS_0L & (
        (data['mTGammaMet']       >   50*1000.) &
        (data['mTGammaMet']       <  100*1000.) &
        (data['met_signif']       >   14      ) &
        (data['dPhiGammaMet']     <  0.7      )
    )

    SR_0L_mT_hgh_loose = PS_0L & (
        (data['mTGammaMet']       >  100*1000.) &
        (data['met_signif']       >   11      ) &
        (data['dPhiGammaMet']     <  1.0      )
    )

    VR_0L_mT_mid = PS_0L & (
        (data['mTGammaMet']       >   50*1000.) &
        (data['mTGammaMet']       <  100*1000.) &
        (data['dPhiGammaMet']     >  2.0      )
    )

    return {
        'Preselection': {'0L':        PS_0L        },
        'SR':           {'0L-mT-low': SR_0L_mT_low,
                         '0L-mT-mid': SR_0L_mT_mid,
                         '0L-mT-hgh': SR_0L_mT_hgh,
                         '0L-mT-low-loose': SR_0L_mT_low_loose,
                         '0L-mT-mid-loose': SR_0L_mT_mid_loose,
                         '0L-mT-hgh-loose': SR_0L_mT_hgh_loose,
                         },
        'VR':           {'0L-mT-mid': VR_0L_mT_mid},
    }


def get_photon_id_masks(data):
    """Return per-event photon ID masks for Tight, LoosePrime4, and Loose working points."""
    baseline  = (data['ph_select_baseline'] == 1)
    tight     = (data['ph_select_tightID']  == 1)
    not_tight = (data['ph_select_tightID']  == 0)
    lp4_isem  = ((data['ph_isEM'] & LoosePrimeDefs['LoosePrime4']) == 0)

    return {
        'Tight':       baseline & tight,
        'LoosePrime4': baseline & not_tight & lp4_isem,
        'Loose':       baseline & not_tight,
    }


def get_truth_masks(data):
    """Return per-event photon truth category masks."""
    return {
        'Real':  (data['ph_truthprompt'] == 1),
        'EFP':   (data['ph_truthEFP']    == 1),
        'JFP':   (data['ph_truthJFP']    == 1),
        'Other': (data['ph_truthother']  == 1),
    }


def fill_iso_histograms(data, event_mask, id_mask, variable, bins, totalweight,
                        norm_variable=None):
    """
    Fill weighted histograms of an isolation variable split by truth category.

    If norm_variable is given (e.g. 'ph_pt'), each event's value is divided by
    data[norm_variable] before histogramming, producing a dimensionless ratio.

    Returns:
        counts: dict mapping truth category -> summed-weight array (len(bins)-1)
        sumw2:  dict mapping truth category -> sum-of-weights-squared array,
                giving per-bin stat uncertainty as sqrt(sumw2[cat])
    """
    truth_masks = get_truth_masks(data)
    combined = event_mask & id_mask

    counts = {}
    sumw2  = {}
    for cat, tmask in truth_masks.items():
        sel = combined & tmask
        w   = totalweight[sel]
        raw = data[variable][sel]
        if norm_variable is not None:
            raw = raw / data[norm_variable][sel]
        values = np.minimum(raw, bins[-1])  # overflow -> last bin
        counts[cat], _ = np.histogram(values, bins=bins, weights=w)
        sumw2[cat],  _ = np.histogram(values, bins=bins, weights=w**2)
    return counts, sumw2


def ABCDresults(data, mask, isMC, ID="tightID", Iso="tightIso", LoosePrime="LoosePrime4"):
    masks = {}

    LoosePrimeMask = LoosePrimeDefs[LoosePrime]

    if Iso == "noIso":
        masks['TT'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==1)
        masks['TL'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==1)
        masks['LT'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==0) & ((data['ph_isEM'] & LoosePrimeMask)==0)
        masks['LL'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==0) & ((data['ph_isEM'] & LoosePrimeMask)==0)
    else:
        masks['TT'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==1) & (data[f'ph_select_{Iso}']==1) & ((data['ph_isEM'] & LoosePrimeMask)==0)
        masks['TL'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==1) & (data[f'ph_select_{Iso}']==0) & ((data['ph_isEM'] & LoosePrimeMask)==0)
        masks['LT'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==0) & (data[f'ph_select_{Iso}']==1) & ((data['ph_isEM'] & LoosePrimeMask)==0)
        masks['LL'] = (data['ph_select_baseline']==1) & (data[f'ph_select_{ID}']==0) & (data[f'ph_select_{Iso}']==0) & ((data['ph_isEM'] & LoosePrimeMask)==0)

    if isMC:
        real_mask  = (data['ph_truthprompt'] == 1)
        jfp_mask   = (data['ph_truthJFP']    == 1)
        efp_mask   = (data['ph_truthEFP']    == 1)
        other_mask = (data['ph_truthother']  == 1)

    if isMC:
        totalweight = data['weight_total']*data['weight_fjvt_effSF']*data['weight_ftag_effSF_GN2v01_Continuous']*data['weight_jvt_effSF']

    results = getemptyresults()
    for b in ['TT', 'TL', 'LT', 'LL']:
        if isMC:
            results[b]['mc']['nevents']              = np.sum(            mask & masks[b])
            results[b]['mc']['sumweights']           = np.sum(totalweight[mask & masks[b]])
            results[b]['real']['nevents']            = np.sum(            mask & masks[b] & real_mask)
            results[b]['real']['sumweights']         = np.sum(totalweight[mask & masks[b] & real_mask])
            results[b]['jfp']['nevents']             = np.sum(            mask & masks[b] & jfp_mask)
            results[b]['jfp']['sumweights']          = np.sum(totalweight[mask & masks[b] & jfp_mask])
            results[b]['efp']['nevents']             = np.sum(            mask & masks[b] & efp_mask)
            results[b]['efp']['sumweights']          = np.sum(totalweight[mask & masks[b] & efp_mask])
            results[b]['other']['nevents']           = np.sum(            mask & masks[b] & other_mask)
            results[b]['other']['sumweights']        = np.sum(totalweight[mask & masks[b] & other_mask])
            results[b]['unclassified']['nevents']    = np.sum(            mask & masks[b] & ~real_mask & ~jfp_mask & ~efp_mask & ~other_mask)
            results[b]['unclassified']['sumweights'] = np.sum(totalweight[mask & masks[b] & ~real_mask & ~jfp_mask & ~efp_mask & ~other_mask])

            for t in ["mc", "real", "jfp", "efp", "other", "unclassified"]:
                if results[b][t]['nevents'] > 0:
                    results[b][t]['staterr'] = np.sqrt((results[b][t]['sumweights']**2)/results[b][t]['nevents'])
        else:
            sel = mask & masks[b]
            n = int(np.sum(sel))
            if 'weight_ph_rprime' in data:
                w    = data['weight_ph_rprime'][sel]
                sw   = float(w.sum())
                stat = float(np.sqrt((w**2).sum())) if n > 0 else 0.0
            else:
                sw   = float(n)
                stat = float(np.sqrt(n)) if n > 0 else 0.0
            results[b]['data'] = {'nevents': n, 'sumweights': sw, 'staterr': stat}
            if n < 100:
                results[b]['runNumbers']   = data['runNumber'][sel]
                results[b]['eventNumbers'] = data['eventNumber'][sel]

    if results[b]['mc']['nevents'] != (results[b]['real']['nevents'] +
                                       results[b]['jfp']['nevents'] +
                                       results[b]['efp']['nevents'] +
                                       results[b]['other']['nevents'] +
                                       results[b]['unclassified']['nevents']):
        print("sums don't match")
    return results


def dumpjson(data, isMC, ID="tightID", Iso="tightIso", LoosePrime="LoosePrime4"):
    regions = get_region_masks(data)
    PS = regions['Preselection']
    SR = regions['SR']
    VR = regions['VR']

    return {'Preselection': {'0L':        ABCDresults(data, PS['0L'],        isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime)},
            'SR':           {'0L-mT-low': ABCDresults(data, SR['0L-mT-low'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime),
                             '0L-mT-mid': ABCDresults(data, SR['0L-mT-mid'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime),
                             '0L-mT-hgh': ABCDresults(data, SR['0L-mT-hgh'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime),
                             '0L-mT-low-loose': ABCDresults(data, SR['0L-mT-low-loose'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime),
                             '0L-mT-mid-loose': ABCDresults(data, SR['0L-mT-mid-loose'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime),
                             '0L-mT-hgh-loose': ABCDresults(data, SR['0L-mT-hgh-loose'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime)},
            'VR':           {'0L-mT-mid': ABCDresults(data, VR['0L-mT-mid'], isMC, ID=ID, Iso=Iso, LoosePrime=LoosePrime)},
           }


def getfakeestimate(regiontype="SR", regionname="0L-mT-low", ID="tightID", Iso="tightIso", LoosePrime="LoosePrime4",
                    Run2=True, debug=False, tag="3.X"):

    totalresults = getemptyresults()
    sample_max = {}
    sample_max['TL'] = [0, '']
    sample_max['LT'] = [0, '']
    sample_max['TT'] = [0, '']
    sample_max['LL'] = [0, '']

    samples = []

    searchpath = f"ABCD_results_{tag}/*{ID}_{Iso}_{LoosePrime}.json"

    for fp in glob.glob(searchpath):
        if "gammajet" in fp: continue
        if "jetjet" in fp: continue
        if "jj" in fp: continue
        if "N2" in fp: continue

        if Run2 and not _is_run2(fp): continue
        elif not Run2 and not _is_run3(fp): continue

        data = load_json_file(fp)

        sample_tag = fp.replace(f"ABCD_results_{tag}/output_", "").replace(".json", "")
        samples.append(sample_tag)

        region = data[regiontype][regionname]

        for b in ['TT', 'TL', 'LT', 'LL']:
            d = region[b]["data"]
            if isinstance(d, dict):
                totalresults[b]['data']['nevents']    += d['nevents']
                totalresults[b]['data']['sumweights'] += d['sumweights']
                totalresults[b]['data']['staterr']    += d['staterr']**2
            else:
                n = int(d)
                totalresults[b]['data']['nevents']    += n
                totalresults[b]['data']['sumweights'] += float(n)
                totalresults[b]['data']['staterr']    += float(n)

            for t in ['real', 'jfp', 'efp', 'other', 'unclassified']:
                for s in ["sumweights", "nevents"]:
                    totalresults[b][t][s] += region[b][t][s]
                if "staterr" in totalresults[b][t] and "staterr" in region[b][t]:
                    totalresults[b][t]["staterr"] += region[b][t]["staterr"]**2.

            if sample_max[b][0] < region[b]["real"]["sumweights"]:
                sample_max[b][0] = region[b]["real"]["sumweights"]
                sample_max[b][1] = sample_tag

    for b in ['TT', 'TL', 'LT', 'LL']:
        totalresults[b]['data']['staterr'] = np.sqrt(totalresults[b]['data']['staterr'])
        for t in ['real', 'jfp', 'efp', 'other', 'unclassified']:
            if "staterr" in totalresults[b][t]:
                totalresults[b][t]["staterr"] = np.sqrt(totalresults[b][t]["staterr"])

    if debug:
        print(json.dumps(totalresults, indent=4, cls=NumpyEncoder))

        print("Most contributing samples:")
        for b in ['TT', 'TL', 'LT', 'LL']:
            print(f"{b}: {sample_max[b][1][:-1]}")

        mcs_data = load_json_file(f"ABCD_results/output_{sample_max['TT'][1]}.json")
        print(json.dumps(mcs_data, indent=4, cls=NumpyEncoder))

    return totalresults, samples


def getsignalestimate(regiontype="SR", regionname="0L-mT-low", ID="tightID", Iso="tightIso", LoosePrime="LoosePrime4",
                      sample="N2_200_N1_190", Run2=True, debug=False, tag="3.X"):
    signalresults = getemptyresults()

    for fp in glob.glob(f"ABCD_results_{tag}/*{sample}*{ID}_{Iso}_{LoosePrime}.json"):
        data = load_json_file(fp)

        if Run2 and not _is_run2(fp): continue
        elif not Run2 and not _is_run3(fp): continue

        region = data[regiontype][regionname]

        for b in ['TT', 'TL', 'LT', 'LL']:
            _d = region[b]["data"]
            if isinstance(_d, dict):
                signalresults[b]['data']['nevents']    += _d['nevents']
                signalresults[b]['data']['sumweights'] += _d['sumweights']
                signalresults[b]['data']['staterr']    += _d['staterr']**2
            else:
                n = int(_d)
                signalresults[b]['data']['nevents']    += n
                signalresults[b]['data']['sumweights'] += float(n)
                signalresults[b]['data']['staterr']    += float(n)
            signalresults[b]['real']['sumweights'] += region[b]["real"]["sumweights"]
            signalresults[b]['jfp']['sumweights'] += region[b]["jfp"]["sumweights"]
            signalresults[b]['efp']['sumweights'] += region[b]["efp"]["sumweights"]
            signalresults[b]['other']['sumweights'] += region[b]["other"]["sumweights"]
            signalresults[b]['unclassified']['sumweights'] += region[b]["unclassified"]["sumweights"]

    for b in ['TT', 'TL', 'LT', 'LL']:
        signalresults[b]['data']['staterr'] = np.sqrt(signalresults[b]['data']['staterr'])

    return signalresults


def yieldsABCD(sresults, useMC=True):
    mininden = 0.1

    N_all = {}
    for b in ['TT', 'TL', 'LT', 'LL']:
        if useMC:
            N_all[b] = sresults[b]['real']['sumweights'] + sresults[b]['jfp']['sumweights'] + sresults[b]['efp']['sumweights'] + sresults[b]['other']['sumweights']
        else:
            d = sresults[b]['data']
            N_all[b] = d['sumweights'] if isinstance(d, dict) else float(d)

    # prompt subtraction
    n_TT = (N_all['TT'] - sresults['TT']['real']['sumweights'] - sresults['TT']['efp']['sumweights'])
    n_TL = (N_all['TL'] - sresults['TL']['real']['sumweights'] - sresults['TL']['efp']['sumweights'])
    n_LT = (N_all['LT'] - sresults['LT']['real']['sumweights'] - sresults['LT']['efp']['sumweights'])
    n_LL = (N_all['LL'] - sresults['LL']['real']['sumweights'] - sresults['LL']['efp']['sumweights'])

    # calculate uncertainty
    d_TT = np.sqrt(abs(N_all['TT']) + (sresults['TT']['real']['staterr']**2) + (sresults['TT']['efp']['staterr']**2))
    d_TL = np.sqrt(abs(N_all['TL']) + (sresults['TL']['real']['staterr']**2) + (sresults['TL']['efp']['staterr']**2))
    d_LT = np.sqrt(abs(N_all['LT']) + (sresults['LT']['real']['staterr']**2) + (sresults['LT']['efp']['staterr']**2))
    d_LL = np.sqrt(abs(N_all['LL']) + (sresults['LL']['real']['staterr']**2) + (sresults['LL']['efp']['staterr']**2))

    # ABCD yield
    N_TT_jfp_est = 0.
    if n_LL > 0:
        N_TT_jfp_est = n_TL * n_LT / n_LL

    # ABCD uncertainty
    d_TT_jfp_est = 0.1
    if n_LL > 0 and N_TT_jfp_est > 0:
        d_TT_jfp_est = abs(N_TT_jfp_est * np.sqrt((d_TL/max(mininden, n_TL))**2 + (d_LT/max(mininden, n_LT))**2 + (d_LL/max(mininden, n_LL))**2))

    Rprime = 0.
    dRprime = 0.1
    if n_LT > 0 and n_TL > 0:
        Rprime = n_TT * n_LL / (n_LT * n_TL)
        dRprime = np.sqrt((d_TT/max(mininden, n_TT))**2 + (d_TL/max(mininden, n_TL))**2 + (d_LT/max(mininden, n_LT))**2 + (d_LL/max(mininden, n_LL))**2)

    return N_TT_jfp_est, d_TT_jfp_est, n_LL, Rprime, dRprime


def printregion(regiontype="SR", region="0L-mT-low", Run2=True, LoosePrime="LoosePrime4", debugoutput=False,
                tag="3.X", sigsamples=None):
    if sigsamples is None:
        sigsamples = DEFAULT_SIGSAMPLES

    blindTT = (regiontype == "SR")

    separator = "-"*128

    era = "Run 2" if Run2 else "Run 3"
    title = f"{regiontype}-{region}  |  {era}  |  {LoosePrime}  |  tag={tag}"
    outer_sep = "=" * 128

    IDs = ["tightID"]
    Isos = ["hybridCOIso"]
    print(outer_sep)
    print(title)
    print("-" * len(title))
    if not debugoutput:
        print(f"      ID    Isolation :   JFP,MC   JFP,DD              {sigsamples[0]}   {sigsamples[1]}   {sigsamples[2]}   {sigsamples[3]}")
        print(separator)

    for ID in IDs:
        for Iso in Isos:
            if debugoutput: print(f"\nResults for {ID}, {Iso}:\n")

            totalresults, samples = getfakeestimate(regiontype, region, ID, Iso, LoosePrime, Run2, False, tag=tag)

            N = {}
            N_MC = {}
            N_JFP_MC = {}
            if debugoutput: print(f"      Data(n) Data(sw)       MC      Real      EFP    Other      JFP")
            for b in ['TT', 'TL', 'LT', 'LL']:
                _d = totalresults[b]['data']
                _d_nevents  = _d['nevents']    if isinstance(_d, dict) else int(_d)
                _d_sw       = _d['sumweights'] if isinstance(_d, dict) else float(_d)
                N[b] = _d_sw - totalresults[b]['real']['sumweights'] - totalresults[b]['efp']['sumweights']
                N_JFP_MC[b] = totalresults[b]['jfp']['sumweights'] + totalresults[b]['other']['sumweights']
                N_MC[b] = N_JFP_MC[b] + totalresults[b]['real']['sumweights'] + totalresults[b]['efp']['sumweights']
                if debugoutput:
                    _show_n  = _d_nevents  if b != 'TT' or (not blindTT) else 0
                    _show_sw = _d_sw       if b != 'TT' or (not blindTT) else 0.0
                    print(f"{b}: {_show_n:8d} {_show_sw:8.1f}   {N_MC[b]:6.1f}    {totalresults[b]['real']['sumweights']:6.1f}   {totalresults[b]['efp']['sumweights']:6.1f}   {totalresults[b]['other']['sumweights']:6.1f}   {totalresults[b]['jfp']['sumweights']:6.1f}")
            if debugoutput: print('')

            N_TT_bkg_DDjfp, U_TT_bkg_DDjfp, den_LL, Rprime, dRprime = yieldsABCD(totalresults, False)

            if N_JFP_MC['LL'] > 0:
                N_TT_bkg_DDMCjfp = N_JFP_MC['TL'] * N_JFP_MC['LT'] / N_JFP_MC['LL']
            else:
                N_TT_bkg_DDMCjfp = 0

            N_TT_bkg_real = totalresults['TT']['real']['sumweights']
            N_TT_bkg_other = totalresults['TT']['other']['sumweights']
            N_TT_bkg_MCjfp = totalresults['TT']['jfp']['sumweights']
            N_TT_bkg_efp = totalresults['TT']['efp']['sumweights']
            N_TT_bkg_unclassified = totalresults['TT']['unclassified']['sumweights']

            N_TT_bkg_MC = N_TT_bkg_MCjfp + N_TT_bkg_real + N_TT_bkg_other + N_TT_bkg_efp

            N_TT_bkg_DD = N_TT_bkg_DDjfp + N_TT_bkg_real + N_TT_bkg_efp

            N_TT_bkg_DDMC = N_TT_bkg_DDMCjfp + N_TT_bkg_real + N_TT_bkg_efp

            if debugoutput: print(f"N_TT_bkg = ({N['TL']:.1f}*{N['LT']:.1f})/({N['LL']:.1f}) = {N_TT_bkg_DDjfp:.1f}")

            if debugoutput: print("")

            if debugoutput:
                if not blindTT:
                    _tt_d = totalresults['TT']['data']
                    _tt_sw = _tt_d['sumweights'] if isinstance(_tt_d, dict) else float(_tt_d)
                    print(f"Total data in TT region is {_tt_sw:.1f}.")
                print(f"DD background prediction: {N_TT_bkg_real:7.1f} (real) + {N_TT_bkg_DDjfp:7.1f} (+/- {U_TT_bkg_DDjfp:7.2f}) (jfp+other) + {N_TT_bkg_efp:.1f} (efp) + {N_TT_bkg_unclassified:.1f} (unclassified) = {N_TT_bkg_DD:.1f}")
                print(f"MC background prediction: {N_TT_bkg_real:7.1f} (real) + {N_TT_bkg_MCjfp+N_TT_bkg_other:7.1f}               (jfp+other) + {N_TT_bkg_efp:.1f} (efp) + {N_TT_bkg_unclassified:.1f} (unclassified) = {N_TT_bkg_MC:.1f}")
                print(f"MC background closure   : {N_TT_bkg_real:7.1f} (real) + {N_TT_bkg_DDMCjfp:7.1f}               (jfp+other) + {N_TT_bkg_efp:.1f} (efp) + {N_TT_bkg_unclassified:.1f} (unclassified) = {N_TT_bkg_DDMC:.1f}")
                print(f"R': {Rprime:7.3f} +/- {dRprime:7.3f}")

            yields = {}
            yieldsstring = ""
            for sigsample in sigsamples:
                sigsampleresults = getsignalestimate(regiontype, region, ID, Iso, LoosePrime, sigsample, Run2, False, tag=tag)
                yields[sigsample] = sigsampleresults['TT']['real']['sumweights']
                if debugoutput: print(f"Signal sample {sigsample} has {yields[sigsample]:.1f} events")
                yieldsstring += f"{yields[sigsample]:13.1f}   "

            if not debugoutput:
                print(f"{ID:10s} {Iso:11s}:   {N_TT_bkg_MCjfp+N_TT_bkg_other:6.1f}    {N_TT_bkg_DDjfp:5.1f} +/- {U_TT_bkg_DDjfp:4.1f}     {yieldsstring}")
            print(separator)
    print("")
    print(outer_sep)


def sampleABCD(sample, debug=False, ID="tightID", Iso="hybridIso", LoosePrime="LoosePrime4a",
               regiontype="SR", region="0L-mT-mid", tag="3.X"):
    sresults = None
    if isinstance(sample, str):
        sname = f"ABCD_results_{tag}/output_{sample}.json"
        sresults = load_json_file(sname)[regiontype][region]
    elif isinstance(sample, dict):
        sresults = sample
    else:
        print("Must provide either valid sample string or dictionary of results.")
        return None

    N_TT_jfp_est, d_TT_jfp_est, den_LL, Rprime, dRprime = yieldsABCD(sresults)

    shortsample = sample.replace(f"_ABCD_{ID}_{Iso}_{LoosePrime}", "")
    sigma = -5.2
    ratio = -0.1
    if debug and den_LL > 0 and sresults['TT']['jfp']['sumweights'] > 0:
        MCJFPtotal = (sresults['TT']['jfp']['sumweights'] + sresults['TT']['other']['sumweights'])
        ratio = (N_TT_jfp_est) / MCJFPtotal
        dratio = (d_TT_jfp_est) / MCJFPtotal
        sigma = (ratio - 1.) / dratio
        print(f"{shortsample:60s} {sresults['TT']['real']['sumweights']:6.1f}  {sresults['TT']['efp']['sumweights']:6.1f}   {sresults['TT']['other']['sumweights']:6.1f}   {sresults['TT']['jfp']['sumweights']:6.1f}  {N_TT_jfp_est:6.1f} +/- {d_TT_jfp_est:4.1f}  {ratio:6.2f} +/- {d_TT_jfp_est/MCJFPtotal:4.2f} = {sigma:4.1f} sigma   {Rprime:5.3f} +/- {dRprime:5.3f}")
    elif debug:
        print(f"{shortsample:60s} {sresults['TT']['real']['sumweights']:6.1f}  {sresults['TT']['efp']['sumweights']:6.1f}   {sresults['TT']['other']['sumweights']:6.1f}   {sresults['TT']['jfp']['sumweights']:6.1f}  {N_TT_jfp_est:6.1f} +/- {d_TT_jfp_est:4.1f}  ")

    return N_TT_jfp_est, sigma, ratio, Rprime, dRprime


def weighted_average(values, errors):
    """
    Compute the weighted average and its uncertainty for uncorrelated measurements.

    Parameters
    ----------
    values : array-like
        The measured values (x_i).
    errors : array-like
        The corresponding 1-sigma uncertainties (σ_i).

    Returns
    -------
    mean : float
        Weighted mean of the values.
    mean_error : float
        Uncertainty of the weighted mean.
    chi2 : float
        Chi-square of the fit (useful for consistency check).
    chi2_ndf : float
        Reduced chi-square (chi2 / (N-1)).
    """
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    weights = 1.0 / errors**2
    mean = np.sum(weights * values) / np.sum(weights)
    mean_error = np.sqrt(1.0 / np.sum(weights))
    chi2 = np.sum(weights * (values - mean)**2)
    chi2_ndf = chi2 / (len(values) - 1) if len(values) > 1 else np.nan

    return mean, mean_error, chi2, chi2_ndf


def printallsamples(ID="tightID", Iso="hybridIso", LoosePrime="LoosePrime4a", Run2=True,
                    regiontype="SR", region="0L-mT-mid", tag="3.X"):
    totalresults, samples = getfakeestimate(regiontype, region, ID, Iso, LoosePrime, Run2, debug=False, tag=tag)
    print(f"{'Run2' if Run2 else 'Run3'} {'Sample':55s} {'Prompt':6s}     {'EFP':6s} {'Other':6s} {'MC JFPs':6s}    {'ABCD':16s}{'ABCD/MC':6s}                       {'Rprime':30s}")
    print("-"*158)

    sigmas = []
    ratios = []
    dratios = []
    Rprimes = []
    dRprimes = []
    for s in sorted(samples):
        est, sigma, ratio, Rprime, dRprime = sampleABCD(s, True, ID, Iso, LoosePrime, regiontype, region, tag=tag)

        if est > 1.:
            if ratio > 0:
                sigmas.append(sigma)
                ratios.append(ratio)
                dratios.append((ratio - 1) / sigma)

            if Rprime > 0:
                Rprimes.append(Rprime)
                dRprimes.append(dRprime)

    print("")

    meanRprime, dmeanRprime, _, _ = weighted_average(Rprimes, dRprimes)
    print(f"Weighted average R'        : {meanRprime:5.3f} +/- {dmeanRprime:5.3f}")

    meanratio, dmeanratio, _, _ = weighted_average(list(1./np.array(ratios)), dratios)
    print(f"Weighted average Truth/ABCD: {meanratio:5.3f} +/- {dmeanratio:5.3f}")

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    ax1.hist(sigmas, bins=50, range=(-5, 5))
    ax1.set_xlabel('ABCD - Truth [$\\sigma$]')
    ax1.set_ylabel('Number of MC Samples')
    ax1.grid(True)

    ax2.hist(ratios, bins=50, range=(0, 3))
    ax2.set_xlabel('ABCD/Truth')
    ax2.set_ylabel('Number of MC Samples')
    ax2.grid(True)

    ax3.hist(Rprimes, bins=50, range=(0, 3))
    ax3.set_xlabel('R\'')
    ax3.set_ylabel('Number of MC Samples')
    ax3.grid(True)

    fig.savefig(f"ABCD_results_{tag}/pulls_{ID}_{Iso}_{LoosePrime}_{'Run2' if Run2 else 'Run3'}_{regiontype}_{region}.pdf")
