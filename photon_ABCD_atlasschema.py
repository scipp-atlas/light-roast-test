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
        SR_selections = {
            "met": (events.met.met < 250 * 1.0e3),
            "lepton_veto": (ak.sum(leptons.pt, axis=1) == 0),
            "leading_jet_pt": (ak.firsts(events.jet.pt) > 100 * 1.0e3),
            "min_dphi_jet_met": (ak.min(abs(events.met.delta_phi(events.jet)), axis=1) > 0.4),
            "bjet_veto": (ak.sum(events.jet.btag_select, axis=1) == 0),
        }

        if isMC:
            SR_selections["vgamma_overlap"] = (events["in"]["vgamma_overlap_7"]==1)
        
        SR_selection = PackedSelection()
        SR_selection.add_multiple(SR_selections)

        SR_presel         = SR_selection.all()
        SR_presel_events  = events[SR_presel]
        SR_presel_entries = ak.num(SR_presel_events,axis=0)
        
        # photon object preselection
        SR_ph_preselection = (
            (SR_presel_events.ph.pt > 10000) &
            (SR_presel_events.ph.select_baseline == 1) &
            ((SR_presel_events.ph.isEM&0x45fc01) == 0) &
            ((abs(SR_presel_events.ph.eta)<1.37) | ((abs(SR_presel_events.ph.eta)>1.52) & 
                                                    (abs(SR_presel_events.ph.eta)<2.37))) &
            (SR_presel_events.ph.select_or_dR02Ph == 1)
        )

        # this selects events with at least one baseline photon
        SR_ph_presel_data = SR_presel_events[ak.any(SR_ph_preselection,axis=1)]
        SR_entries        = ak.num(SR_ph_presel_data,axis=0)
        
        # define tight and loose cuts, now on the smaller data sample that only has good events
        SR_ph_selection=((SR_ph_presel_data.ph.pt>10000) & 
                         ((abs(SR_ph_presel_data.ph.eta)<1.37) | ((abs(SR_ph_presel_data.ph.eta)>1.52) & 
                                                                  (abs(SR_ph_presel_data.ph.eta)<2.37))) &
                         (SR_ph_presel_data.ph.select_or_dR02Ph==1) &
                         ((SR_ph_presel_data.ph.isEM&0x45fc01)==0) &
                         (SR_ph_presel_data.ph.select_baseline==1)
                        )      

        ABCD=None
        if ak.num(SR_ph_selection,axis=0)>0 and SR_entries>0:
            # get the index of the first preselected photon (which should be the leading preselected photon)
            indices=ak.argmax(SR_ph_selection,axis=1,keepdims=True)
            
            # apply cuts to that index
            ph_tight = (ak.firsts(SR_ph_presel_data.ph[indices].select_tightID)==1)
            ph_iso   = (ak.firsts(SR_ph_presel_data.ph[indices].select_tightIso)==1)
    
            if isMC:
                ph_truth = ((ak.firsts(SR_ph_presel_data.ph[indices].truthType) != 0) &
                            (ak.firsts(SR_ph_presel_data.ph[indices].truthType) != 16))
                ABCD={
                    "A_true": ak.num(SR_ph_presel_data.ph[indices].pt[ ph_tight & ~ph_iso &  ph_truth][:,0],axis=0),
                    "B_true": ak.num(SR_ph_presel_data.ph[indices].pt[~ph_tight & ~ph_iso &  ph_truth][:,0],axis=0),
                    "C_true": ak.num(SR_ph_presel_data.ph[indices].pt[ ph_tight &  ph_iso &  ph_truth][:,0],axis=0),
                    "D_true": ak.num(SR_ph_presel_data.ph[indices].pt[~ph_tight &  ph_iso &  ph_truth][:,0],axis=0),
                    "A_fake": ak.num(SR_ph_presel_data.ph[indices].pt[ ph_tight & ~ph_iso & ~ph_truth][:,0],axis=0),
                    "B_fake": ak.num(SR_ph_presel_data.ph[indices].pt[~ph_tight & ~ph_iso & ~ph_truth][:,0],axis=0),
                    "C_fake": ak.num(SR_ph_presel_data.ph[indices].pt[ ph_tight &  ph_iso & ~ph_truth][:,0],axis=0),
                    "D_fake": ak.num(SR_ph_presel_data.ph[indices].pt[~ph_tight &  ph_iso & ~ph_truth][:,0],axis=0),
                }
            else:
                ABCD = {
                    "A_data": ak.num(SR_ph_presel_data.ph[indices].pt[ ph_tight & ~ph_iso][:,0],axis=0),
                    "B_data": ak.num(SR_ph_presel_data.ph[indices].pt[~ph_tight & ~ph_iso][:,0],axis=0),
                    "C_data": ak.num(SR_ph_presel_data.ph[indices].pt[ ph_tight &  ph_iso][:,0],axis=0) if unblindSR else 0.,
                    "D_data": ak.num(SR_ph_presel_data.ph[indices].pt[~ph_tight &  ph_iso][:,0],axis=0),
                }
        else:
            ABCD={"ak.num(SR_ph_selection,axis=0)": ak.num(SR_ph_selection,axis=0),
                 "ak.argmax(SR_ph_selection,axis=1)": ak.argmax(SR_ph_selection,axis=1)}

        return {
            "total": {
                "entries": ak.num(events, axis=0)
            },
            "presel": {
                "entries": SR_presel_entries,
            },
            "SR":{
                "entries": SR_entries,
            },
            "ABCD": ABCD
        }

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":
    start_time = time.time()
    
    cluster=None
    dataset=None
    can_submit_to_condor=True
    
    if can_submit_to_condor:
        # To facilitate usage with HTCondor
        cluster = HTCondorCluster(
            log_directory=Path().cwd() / ".condor_logs" / "cutflows_v2",
            cores=4,
            memory="4GB",
            disk="2GB",
            #silence_logs="debug",
        )
        cluster.scale(jobs=100)
    
        # if we're running over all samples, ensure that here
        #inputfiles="af_v2_2.json"      # data+MC
        inputfiles="af_v2_2_mc.json"   # MC
        #inputfiles="af_v2_2_data.json" # data
        
        dataset = json.loads(Path(inputfiles).read_text())
        
        #datasettag="Znunugamma"
        #datasettag="Wenu_BFilter"
        #dataset = {datasettag: dataset[datasettag]}

    else:
        cluster=LocalCluster()
        
        #inputfiles="af_v2_2_mc_onefile.json"   # MC
        inputfiles="af_v2_2_mc.json"
        #inputfiles="af_v2_2_data_onefile.json" # data

        dataset = json.loads(Path(inputfiles).read_text())

        if True:
            datasettag="Wenu_BFilter" #"Znunugamma" #"Wenu_BFilter"
            dataset = {datasettag: dataset[datasettag]}

            if False:
                nfiles=len(list(dataset[datasettag]["files"].keys()))
                print(nfiles)
                filestart=20
                fileend=21
                filecount=0
                for k in list(dataset[datasettag]["files"].keys()):
                    if filecount < filestart or filecount >= fileend:
                        del dataset[datasettag]["files"][k]
                    filecount+=1
                

        
    client = Client(cluster)
    
    print("Applying to fileset")
    my_processor = MyProcessor()    
    out = apply_to_fileset(
        my_processor,
        dataset,
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