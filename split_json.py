import json
from pathlib import Path

version="2_2"

dataset = json.loads(Path(f"af_v{version}.json").read_text())
dataset_mc = dataset.copy()
dataset_data = dataset.copy()

for k in list(dataset.keys()):
    if "data" in k:
        del dataset_mc[k]
    else:
        del dataset_data[k]

with open(f"af_v{version}_mc.json",'w') as fp:
    json.dump(dataset_mc, fp, indent=4)

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
