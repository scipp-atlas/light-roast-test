import json
from pathlib import Path

version="3"

dataset = json.loads(Path(f"/data/kratsg/radiative-decays/af_v{version}.json").read_text())
dataset_mc = dataset.copy()
dataset_sig = dataset.copy()
dataset_data = dataset.copy()

for k in list(dataset.keys()):
    if "data" in k:
        del dataset_mc[k]
        del dataset_sig[k]
    elif "N2_" in k:
        del dataset_mc[k]
        del dataset_data[k]
    else:
        del dataset_data[k]
        del dataset_sig[k]

with open(f"af_v{version}_mc.json",'w') as fp:
    json.dump(dataset_mc, fp, indent=4)

with open(f"af_v{version}_sig.json",'w') as fp:
    json.dump(dataset_sig, fp, indent=4)

with open(f"af_v{version}_data.json",'w') as fp:
    json.dump(dataset_data, fp, indent=4)


# onefile versions
mc_sample="Znunugamma"
for k in list(dataset_mc.keys()):
    if k != mc_sample:
        del dataset_mc[k]

firstfile=True
for k in list(dataset_mc[mc_sample]["files"].keys()):
    if firstfile:
        firstfile=False
        continue
    else:
        del dataset_mc[mc_sample]["files"][k]

with open(f"af_v{version}_mc_onefile.json",'w') as fp:
    json.dump(dataset_mc, fp, indent=4)


data_sample="data_2017"
for k in list(dataset_data.keys()):
    if k != data_sample:
        del dataset_data[k]

filename="42164743._000036"
for k in list(dataset_data[data_sample]["files"].keys()):
    if filename in k: continue
    else: del dataset_data[data_sample]["files"][k]

with open(f"af_v{version}_data_onefile.json",'w') as fp:
    json.dump(dataset_data, fp, indent=4)
