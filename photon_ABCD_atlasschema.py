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
import uproot

import coffea

from atlas_schema.methods import behavior as as_behavior
from atlas_schema.schema import NtupleSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.dataset_tools import apply_to_fileset
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
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
        self._accumulator = {
            "ntuple": {
                "ph_pt": [],
                "ph_eta": [],
                "ph_select_tightID": [],
                "ph_isEM": [],
                "ph_select_tightIso": [],
                "ph_truthType": [],
                "ph_truthOrigin": [],
                "met_met": [],
                "met_phi": []
            }
        }

    @property
    def accumulator(self):
        return self._accumulator
        
    def process(self, events):
        
        output = self.accumulator
        
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

        # Extract some event-level data
        output["ntuple"]["ph_pt"].append(SR_ph_presel_data.ph[indices].pt)
        output["ntuple"]["ph_eta"].append(SR_ph_presel_data.ph[indices].eta)
        output["ntuple"]["ph_select_tightID"].append(SR_ph_presel_data.ph[indices].select_tightID)
        output["ntuple"]["ph_isEM"].append(SR_ph_presel_data.ph[indices].isEM)
        output["ntuple"]["ph_select_tightIso"].append(SR_ph_presel_data.ph[indices].select_tightIso)
        output["ntuple"]["ph_truthType"].append(SR_ph_presel_data.ph[indices].truthType)
        output["ntuple"]["ph_truthOrigin"].append(SR_ph_presel_data.ph[indices].truthOrigin)
        output["ntuple"]["met_met"].append(SR_ph_presel_data.met.met)
        output["ntuple"]["met_phi"].append(SR_ph_presel_data.met.phi)
        
        output["total"]  = ak.num(events, axis=0)
        output["presel"] = SR_presel_entries
        output["SR"]     = SR_entries
        output["ABCD"]   = ABCD

        return output

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":

    # job configuration
    can_submit_to_condor=False

    # ---------------------------------------------------------------------------
    cluster=None
    dataset=None

    # Set this up first to catch obvious problems
    my_processor = MyProcessor()    

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

        inputfiles="af_v2_2_mc_onefile.json"   # MC
        #inputfiles="af_v2_2_mc.json"
        #inputfiles="af_v2_2_data_onefile.json" # data

        dataset = json.loads(Path(inputfiles).read_text())

        if not "onefile" in inputfiles:
            datasettag="Znunugamma" #"Wenu_BFilter"
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
    # ---------------------------------------------------------------------------
    
    # ---------------------------------------------------------------------------
    # construct and run the job
    print("Applying to fileset")
    out = apply_to_fileset(
        my_processor,
        dataset,
        schemaclass=NtupleSchema,
    )
    
    print("Beginning of dask.compute()")  
    start_time = time.time()
    
    (computed,) = dask.compute(out)
    
    end_time = time.time()
    print("Execution time: ", end_time - start_time)
    print("Finished dask.compute")
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Write ntuples out
    for sample in list(computed.keys()):
        print(len(computed[sample]["ntuple"]["ph_pt"]))
              
        data = {}
        for var in list(computed[sample]["ntuple"].keys()):
            data[var] = computed[sample]["ntuple"][var][0].to_numpy()
        
        # Write to a ROOT file using uproot
        with uproot.recreate(f"output_{sample}.root") as f:
            f["mytree"] = data  # uproot can write directly from Awkward arrays

        # remove ntuple before serializing everything else
        del computed[sample]["ntuple"]
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # write out JSON summaries for ABCD calculation
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
    # ---------------------------------------------------------------------------
