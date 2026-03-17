#!/usr/bin/env python3
"""
Terminal-runnable version of photon_ABCD_picontuples.ipynb.
Runs the full ABCD background estimation analysis:
  1. Process ROOT ntuples -> JSON result files
  2. Print ABCD background estimates for SR/VR/Preselection regions
  3. Run per-sample closure tests and save pull plots to PDF
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for terminal use

from abcd_utils import *

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

tag = "4"
base_path = f"/data/mhance/SUSY/ntuples/v{tag}"

sigsamples = [
    "N2_200_N1_185_WB",
    "N2_200_N1_190_WB",
    "N2_200_N1_195_WB",
    "N2_200_N1_197_WB",
]

IDs = ["tightID"]
Isos = ["hybridCOIso"]
LoosePrimes = ["LoosePrime4", "Loose"]

LoosePrime = "LoosePrime4"

# ---------------------------------------------------------------------------
# Step 1: Process ROOT ntuples -> JSON files
# ---------------------------------------------------------------------------

try:
    os.mkdir(f"ABCD_results_{tag}")
except:
    pass

if False:
    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.endswith('.root'): continue
            if "a.root" in file or "d.root" in file or "e.root" in file: continue
            
            filepath = os.path.join(root, file)
            if "output_" not in filepath: continue

            with uproot.open(filepath) as uf:
                if 'picontuple' in uf:
                    tree = uf['picontuple']
                    data = tree.arrays(library="np")
                    
                    for ID in IDs:
                        for Iso in Isos:
                            for LP in LoosePrimes:
                                results = dumpjson(data, "data_" not in filepath, ID, Iso, LP)
                                outfile = f"ABCD_results_{tag}/" + file.replace(".root", f"_ABCD_{ID}_{Iso}_{LP}.json")
                                with open(outfile, 'w') as jf:
                                    json.dump(results, jf, indent=4, cls=NumpyEncoder)

# ---------------------------------------------------------------------------
# Step 2: Print ABCD background estimates
# ---------------------------------------------------------------------------

for region in ["0L-mT-low", "0L-mT-mid", "0L-mT-hgh"]:
    for Run2 in [True, False]:
        printregion("SR", region, Run2, LoosePrime, tag=tag, sigsamples=sigsamples)

for region in ["0L-mT-low"]:
    printregion("SR", region, True , LoosePrime, debugoutput=True, tag=tag, sigsamples=sigsamples)
    printregion("SR", region, False, LoosePrime, debugoutput=True, tag=tag, sigsamples=sigsamples)

for Run2 in [True, False]:
    printregion("VR", "0L-mT-mid", Run2, LoosePrime, debugoutput=False, tag=tag, sigsamples=sigsamples)
    printregion("VR", "0L-mT-mid", Run2, LoosePrime, debugoutput=True , tag=tag, sigsamples=sigsamples)

for Run2 in [True, False]:
    printregion("Preselection", "0L", Run2, LoosePrime, debugoutput=True, tag=tag, sigsamples=sigsamples)

# Loose definition comparison
for Run2 in [True, False]:
    printregion("VR", "0L-mT-mid", Run2, "Loose", tag=tag, sigsamples=sigsamples)
    printregion("VR", "0L-mT-mid", Run2, "Loose", debugoutput=True, tag=tag, sigsamples=sigsamples)

# ---------------------------------------------------------------------------
# Step 3: Per-sample closure tests (saves pull plots to PDF)
# ---------------------------------------------------------------------------


if False:
    ID = "tightID"
    Iso = "hybridCOIso"
    LoosePrime = "LoosePrime4"
    
    for regiontype, region in zip(["VR", "Preselection"], ["0L-mT-mid", "0L"]):
        print("="*200)
        print(f"{regiontype}: {region}")
        print("-"*175)
        for Run2 in [True, False]:
            printallsamples(ID, Iso, LoosePrime, Run2, regiontype, region, tag=tag)
            plt.close('all')
            print("")
        print("="*200)
