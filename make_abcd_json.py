#!/usr/bin/env python3
"""
Process ROOT picontuples -> per-sample ABCD JSON result files.

Walks the ntuple directory, reads each picontuple tree, fills the ABCD
regions, and writes one JSON file per (sample, ID, Iso, LoosePrime)
combination into ABCD_results_<tag>/.
"""

import json
import os

import uproot

from abcd_utils import dumpjson, NumpyEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

tag = "4.3"
base_path = f"/data/mhance/SUSY/ntuples/v{tag}"

file_prefix = "PICOPROD_RAv4"

IDs         = ["tightID"]
Isos        = ["hybridCOIso"]
LoosePrimes = ["LoosePrime4", "Loose"]

# ---------------------------------------------------------------------------
# Process ROOT ntuples -> JSON files
# ---------------------------------------------------------------------------

os.makedirs(f"ABCD_results_{tag}", exist_ok=True)

for root, _, files in os.walk(base_path):
    for file in files:
        if not file.endswith('.root'):
            continue
        if "a.root" in file or "d.root" in file or "e.root" in file:
            continue

        filepath = os.path.join(root, file)
        if f"{file_prefix}_" not in filepath:
            continue
        if "campaigns" in filepath:
            continue

        print(f"processing file {filepath}")

        with uproot.open(filepath) as uf:
            if 'picontuple' not in uf:
                continue
            data = uf['picontuple'].arrays(library="np")

            for ID in IDs:
                for Iso in Isos:
                    for LP in LoosePrimes:
                        results = dumpjson(data, "data_" not in filepath, ID, Iso, LP)
                        outfile = (f"ABCD_results_{tag}/"
                                   + file.replace(".root", f"_ABCD_{ID}_{Iso}_{LP}.json"))
                        with open(outfile, 'w') as jf:
                            json.dump(results, jf, indent=4, cls=NumpyEncoder)
