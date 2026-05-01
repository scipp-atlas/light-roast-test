#!/usr/bin/env python3
"""
Print ABCD background estimates from pre-computed JSON result files.

Prints data-driven JFP estimates, MC JFP yields, and signal yields for
SR, VR, and Preselection regions. Optionally runs per-sample closure
tests and saves pull plots to PDF (set the if-False block to if-True).

Reads JSON files from ABCD_results_<tag>/ (produced by run_abcd.py).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from abcd_utils import printregion, printallsamples

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

tag = "4.3"

sigsamples = [
    "N2_200_N1_185_WB",
    "N2_200_N1_190_WB",
    "N2_200_N1_195_WB",
    "N2_200_N1_197_WB",
]

LoosePrime = "LoosePrime4"

# ---------------------------------------------------------------------------
# Print ABCD background estimates
# ---------------------------------------------------------------------------

for region in ["0L-mT-low", "0L-mT-mid", "0L-mT-hgh"]:
    for Run2 in [True, False]:
        printregion("SR", region, Run2, LoosePrime, tag=tag, sigsamples=sigsamples)

for region in ["0L-mT-low"]:
    printregion("SR", region, True,  LoosePrime, debugoutput=True, tag=tag, sigsamples=sigsamples)
    printregion("SR", region, False, LoosePrime, debugoutput=True, tag=tag, sigsamples=sigsamples)

for Run2 in [True, False]:
    printregion("VR", "0L-mT-mid", Run2, LoosePrime, debugoutput=False, tag=tag, sigsamples=sigsamples)
    printregion("VR", "0L-mT-mid", Run2, LoosePrime, debugoutput=True,  tag=tag, sigsamples=sigsamples)

for Run2 in [True, False]:
    printregion("Preselection", "0L", Run2, LoosePrime, debugoutput=True, tag=tag, sigsamples=sigsamples)

# Loose definition comparison
for Run2 in [True, False]:
    printregion("VR", "0L-mT-mid", Run2, "Loose", tag=tag, sigsamples=sigsamples)
    printregion("VR", "0L-mT-mid", Run2, "Loose", debugoutput=True, tag=tag, sigsamples=sigsamples)

# ---------------------------------------------------------------------------
# Per-sample closure tests (saves pull plots to PDF)
# ---------------------------------------------------------------------------

if False:
    ID         = "tightID"
    Iso        = "hybridCOIso"
    LoosePrime = "LoosePrime4"

    for regiontype, region in zip(["VR", "Preselection"], ["0L-mT-mid", "0L"]):
        print("=" * 200)
        print(f"{regiontype}: {region}")
        print("-" * 175)
        for Run2 in [True, False]:
            printallsamples(ID, Iso, LoosePrime, Run2, regiontype, region, tag=tag)
            plt.close('all')
            print("")
        print("=" * 200)
