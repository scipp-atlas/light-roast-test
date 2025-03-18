# this portion is done to ignore warnings from coffea for now
from __future__ import annotations

import os
import json
import time
from pathlib import Path

import awkward as ak
import dask
import dask_awkward as dak
import parse
import atlas_schema
import numpy as np

import coffea

from atlas_schema.methods import behavior as as_behavior
from atlas_schema.schema import NtupleSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.dataset_tools import apply_to_fileset
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue.htcondor import HTCondorCluster
from dask.distributed import LocalCluster
from matplotlib import pyplot as plt
import hist.dask as had

fname_pattern = parse.compile(
    "user.{username:w}.{dsid:d}.{process:S}.{campaign:w}.v{version:.1f}_ANALYSIS.root"
)

colors_dict = {
    "Znunu": "b",
    "Wenu": "g",
    "Wmunu": "r",
    "Wtaunu_L": "c",
    "Wtaunu_H": "m",
    "Znunugamma": "y",
    "Wmunugamma": "k",
    "Wenugamma": "brown",
    "Wtaunugamma": "pink",
    "N2_100_N1_97_WB_signal": "rosybrown",
    "Fake/Nonprompt": "lime",
}  #  'slategrey', 'blueviolet', 'crimson'


import warnings
warnings.filterwarnings("ignore", module="coffea.*")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        # can define histograms here
        pass

    def process(self, events):
        ## TODO: remove this temporary fix when https://github.com/scikit-hep/vector/issues/498 is resolved
        met_dict = {field: events.met[field] for field in events.met.fields}
        met_dict["pt"] = dak.zeros_like(events.met.met)
        met_dict["eta"] = dak.zeros_like(events.met.met)
        events["met"] = dak.zip(met_dict, with_name="MissingET", behavior=as_behavior)

        dataset = events.metadata["dataset"]
        
        print(f"processing {len(events)} events for {dataset}")
        # xs = events.metadata["xs"]
        # lum = events.metadata["luminosity"]
        # process = events.metadata["process"]
        # genFiltEff = events.metadata["genFiltEff"]
        # evt_count = ak.num(events, axis=0).compute()
        # weights = (xs * genFiltEff * lum / evt_count) * np.ones(evt_count)

        isMC = not "data" in dataset
        unblindSR = False
        
        leptons = ak.concatenate((events.el, events.mu), axis=1)

        # here are some selection cuts for something that looks like the signal region.
        # the only thing that's different is the MET requirement, which I inverted to be
        # met<250 instead of met>250, to make sure we don't accidentally unblind the SR
        selections = {
            "met": (events.met.met > 250 * 1.0e3),
            "lepton_veto": (ak.sum(leptons.pt, axis=1) == 0),
            "leading_jet_pt": (ak.firsts(events.jet.pt) > 100 * 1.0e3),
            "min_dphi_jet_met": (ak.min(abs(events.met.delta_phi(events.jet)), axis=1) > 0.4),
            "bjet_veto": (ak.sum(events.jet.btag_select, axis=1) == 0),
        }

        if isMC:
            selections["vgamma_overlap"] = (events["in"]["vgamma_overlap_7"]==1)
        
        selection = PackedSelection()
        selection.add_multiple(selections)

        SR=(selection.all())
        presel_events=events[SR]
        
        # photon object preselection
        ph_preselection = (
            (presel_events.ph.pt > 10000) &
            (presel_events.ph.select_baseline == 1) &
            ((presel_events.ph.isEM&0x45fc01) == 0) &
            (
             (abs(presel_events.ph.eta)<1.37) | ((abs(presel_events.ph.eta)>1.52) & 
                                                 (abs(presel_events.ph.eta)<2.37))
            ) &
            (presel_events.ph.select_or_dR02Ph == 1)
        )

        # this selects events with at least one baseline photon
        ph_presel_data=presel_events[ak.any(ph_preselection,axis=1)]

        # define tight and loose cuts, now on the smaller data sample that only has good events
        ph_preselection=((ph_presel_data.ph.pt>10000) & 
                         (
                             (abs(ph_presel_data.ph.eta)<1.37) | ((abs(ph_presel_data.ph.eta)>1.52) & 
                                                                  (abs(ph_presel_data.ph.eta)<2.37))
                         ) &
                         (ph_presel_data.ph.select_or_dR02Ph==1) &
                         ((ph_presel_data.ph.isEM&0x45fc01)==0) &
                         (ph_presel_data.ph.select_baseline==1)
                        )

        # get the index of the first preselected photon (which should be the leading preselected photon)
        indices=ak.unflatten(ak.argmax(ph_preselection,axis=1),1)
        
        # apply cuts to that index
        ph_tight = (ak.firsts(ph_presel_data.ph[indices].select_tightID)==1)
        ph_iso   = (ak.firsts(ph_presel_data.ph[indices].select_tightIso)==1)

        ABCD=None
        if isMC:
            ph_truth = ((ak.firsts(ph_presel_data.ph[indices].truthType) != 0) &
                        (ak.firsts(ph_presel_data.ph[indices].truthType) != 16))
            ABCD={
                "A_true": ak.num(ph_presel_data.ph[indices].pt[ ph_tight & ~ph_iso &  ph_truth][:,0],axis=0),
                "B_true": ak.num(ph_presel_data.ph[indices].pt[~ph_tight & ~ph_iso &  ph_truth][:,0],axis=0),
                "C_true": ak.num(ph_presel_data.ph[indices].pt[ ph_tight &  ph_iso &  ph_truth][:,0],axis=0),
                "D_true": ak.num(ph_presel_data.ph[indices].pt[~ph_tight &  ph_iso &  ph_truth][:,0],axis=0),
                "A_fake": ak.num(ph_presel_data.ph[indices].pt[ ph_tight & ~ph_iso & ~ph_truth][:,0],axis=0),
                "B_fake": ak.num(ph_presel_data.ph[indices].pt[~ph_tight & ~ph_iso & ~ph_truth][:,0],axis=0),
                "C_fake": ak.num(ph_presel_data.ph[indices].pt[ ph_tight &  ph_iso & ~ph_truth][:,0],axis=0),
                "D_fake": ak.num(ph_presel_data.ph[indices].pt[~ph_tight &  ph_iso & ~ph_truth][:,0],axis=0),
            }
        else:
            ABCD = {
                "A_data": ak.num(ph_presel_data.ph[indices].pt[ ph_tight & ~ph_iso][:,0],axis=0),
                "B_data": ak.num(ph_presel_data.ph[indices].pt[~ph_tight & ~ph_iso][:,0],axis=0),
                "C_data": ak.num(ph_presel_data.ph[indices].pt[ ph_tight &  ph_iso][:,0],axis=0) if unblindSR else 0.,
                "D_data": ak.num(ph_presel_data.ph[indices].pt[~ph_tight &  ph_iso][:,0],axis=0),
            }
            
        return {
            "total": {
                "entries": ak.num(events, axis=0)
            },
            "presel": {
                "total": ak.num(presel_events,axis=0),
            },
            "ABCD": ABCD
        }

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":
    start_time = time.time()
    
    cluster=None
    dataset_to_run=None
    can_submit_to_condor=True
    
    if can_submit_to_condor:
        # To facilitate usage with HTCondor
        cluster = HTCondorCluster(
            log_directory=Path().cwd() / ".condor_logs" / "cutflows_v2",
            cores=4,
            memory="4GB",
            disk="2GB",
        )
        cluster.scale(jobs=100)
    
        # if we're running over all samples, ensure that here
        dataset_runnable = json.loads(Path("af_v2.json").read_text())
        dataset_to_run=dataset_runnable
    else:
        cluster=LocalCluster()
        dataset_runnable = json.loads(Path("af_v2_onefile.json").read_text())
        datasettag='Znunugamma'
        dataset_to_run={datasettag: dataset_runnable[datasettag]}
    
    
    client = Client(cluster)
    
    print("Applying to fileset")
    my_processor = MyProcessor()    
    out = apply_to_fileset(
        my_processor,
        dataset_to_run,
        schemaclass=NtupleSchema,
    )
    
    print("Beginning of dask.compute()")
    
    # Add progress bar for dask
    pbar = ProgressBar()
    pbar.register()
    
    (computed,) = dask.compute(out)
    end_time = time.time()
    
    print("Execution time: ", end_time - start_time)
    print("Finished dask.compute")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    print(json.dumps(computed,indent=4, cls=NpEncoder))
    with open('results.json', 'w') as fp:
        json.dump(computed, fp, indent=4, cls=NpEncoder)