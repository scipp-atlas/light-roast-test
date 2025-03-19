"""
Preprocess the files and generate a dataset runnable JSON that can be reused.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import parse
from coffea.dataset_tools import filter_files, preprocess
from dask.distributed import Client
from dask_jobqueue.htcondor import HTCondorCluster

fname_pattern_mc = parse.compile(
    "user.{username:w}.{dsid:d}.{process:S}.{campaign:w}.v{version:.1f}_ANALYSIS.root"
)
fname_pattern_data = parse.compile(
    "user.{username:w}.data{year:d}_AllYear.v{version:.1f}_ANALYSIS.root"
)

if __name__ == "__main__":
    start_time = time.time()

    # cross-sections in fb
    xsecs = {
        "545759": 22670.1,
        "545760": 22670.1,
        "545761": 22670.1,
        "545762": 22670.1,
        "545763": 22670.1,
        "545764": 22670.1,
        "545765": 1807.39,
        "545766": 1807.39,
        "545767": 1807.39,
        "545768": 1807.39,
        "545769": 1807.39,
        "545770": 1807.39,
        "545771": 121.013,
        "545772": 121.013,
        "545773": 121.013,
        "545774": 121.013,
        "545775": 121.013,
        "545776": 121.013,
        "545777": 545.57,
        "700335": 447130.00000000005,
        "700336": 447130.00000000005,
        "700337": 447130.00000000005,
        "700338": 21742000.0,
        "700339": 21742000,
        "700340": 21742000,
        "700341": 21806000,
        "700342": 21806000,
        "700343": 21806000,
        "700344": 7680000,
        "700345": 7680000,
        "700346": 7680000,
        "700347": 14126000,
        "700348": 14126000,
        "700349": 14126000,
        "700401": 56438.0,
        "700402": 364840.0,
        "700403": 364830.0,
        "700404": 364840.0,
    }

    filter_effs = {
        "545759": 5.029841e-02,
        "5457560": 5.091961e-02,
        "545761": 5.187749e-02,
        "545762": 5.713781e-02,
        "545763": 1.138526e-01,
        "545764": 1.513492e-01,
        "545765": 9.503700e-02,
        "545766": 9.814262e-02,
        "545767": 1.022704e-01,
        "545768": 1.087524e-01,
        "545769": 1.784614e-01,
        "545770": 2.310872e-01,
        "545771": 1.461057e-01,
        "545772": 1.531744e-01,
        "545773": 1.599781e-01,
        "545774": 1.677462e-01,
        "545775": 2.452122e-01,
        "545776": 3.095950e-01,
        "545777": 1.087590e-01,
        "700335": 8.426931e-02,
        "700336": 0.202381,
        "700337": 7.127170e-01,
        "700338": 9.376371e-03,
        "700339": 0.1489766,
        "700340": 8.435958e-01,
        "700341": 0.0097968,
        "700342": 0.1460112,
        "700343": 8.435538e-01,
        "700344": 9.011307e-03,
        "700345": 0.146033,
        "700346": 8.474706e-01,
        "700347": 0.00867775,
        "700348": 1.433724e-01,
        "700349": 8.474573e-01,
        "700401": 1.000000e00,
        "700402": 1.000000e00,
        "700403": 1.000000e00,
        "700404": 1.000000e00,
    }

    luminosity = 140  # fb-1

    datasets_bkg = Path("/data/maclwong/Ben_Bkg_Samples/v2_2/").glob("*ANALYSIS.root")
    datasets_data = Path("/data/kratsg/radiative-decays/").glob("*v2*ANALYSIS.root")

    fileset = {}

    for dataset in datasets_bkg:
        parsed = fname_pattern_mc.parse(dataset.name)

        if not parsed:
            print(f"Could not parse {dataset.name}, skipping")
            continue

        process = parsed["process"]
        dsid = str(parsed["dsid"])

        if dsid not in xsecs:
            print(f"Missing {dsid} ({process}) in xsecs, skipping")
            continue

        if dsid not in filter_effs:
            print(f"Missing {dsid} ({process}) in filter_effs, skipping")
            continue

        files = {fname: "analysis" for fname in dataset.glob("*.root")}

        if process in fileset:
            fileset[process]["files"].update(files)
        else:
            fileset[process] = {
                "files": files,
                "metadata": {
                    "process": process,
                    "xs": xsecs[dsid],
                    "genFiltEff": filter_effs[dsid],
                    "luminosity": luminosity,
                },
            }

    for dataset in datasets_data:
        parsed = fname_pattern_data.parse(dataset.name)

        if not parsed:
            print(f"Could not parse {dataset.name}, skipping")
            continue

        process = f"data_{parsed['year']}"

        files = {fname: "analysis" for fname in dataset.glob("*.root")}

        if process in fileset:
            fileset[process]["files"].update(files)
        else:
            fileset[process] = {
                "files": files,
                "metadata": {
                    "process": process,
                    "xs": 1.0,
                    "genFiltEff": 1.0,
                    "luminosity": luminosity,
                },
            }

    cluster = HTCondorCluster(
        log_directory=Path().cwd() / ".condor_logs" / "preprocess_v2",
        cores=4,
        memory="4GB",
        disk="2GB",
    )
    cluster.scale(jobs=2 * len(fileset))
    client = Client(cluster)

    dataset_runnable, dataset_updated = preprocess(
        fileset,
        align_clusters=False,
        step_size=50_000,
        files_per_batch=5,
        skip_bad_files=True,
        save_form=False,
        # scheduler=client,
    )

    dataset_runnable_filtered = filter_files(dataset_runnable)

    end_time = time.time()

    print("Execution time: ", end_time - start_time)

    output_path = Path("dataset_runnable")
    output_path.mkdir(exist_ok=True)
    (output_path / "af_v2_2.json").write_text(
        json.dumps(dataset_runnable_filtered, sort_keys=True, indent=4)
    )
