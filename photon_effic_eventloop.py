import ROOT as r
import json
from pathlib import Path
import glob
import os

def processfile(filetag):
    filename=glob.glob(f"/data/mhance/SUSY/ntuples/v3_6/user.bhodkins.RadiativeDecays.{filetag}.v3.*__NOFILTER_ANALYSIS.root/user.bhodkins.*.ANALYSIS.root")
    
    f=r.TFile(filename[0],"RO")
    t=f.Get("analysis")
    
    output = r.TFile(f"efficoutputs/{filetag}.root","RECREATE")

    # 1D plots of pT
    ph_pt_truth = r.TH1F("ph_pt_truth", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_reco = r.TH1F("ph_pt_reco", "ph_pT;truth #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_baseline = r.TH1F("ph_pt_baseline", "ph_pT;truth #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_tightID = r.TH1F("ph_pt_tightID", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID = r.TH1F("ph_pt_mediumID", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_tightIso = r.TH1F("ph_pt_tightIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_looseIso = r.TH1F("ph_pt_looseIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_tightID_tightIso = r.TH1F("ph_pt_tightID_tightIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_tightID_looseIso = r.TH1F("ph_pt_tightID_looseIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_mediumID_tightIso = r.TH1F("ph_pt_mediumID_tightIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID_looseIso = r.TH1F("ph_pt_mediumID_looseIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    # 1D plots of isolation after medium or tight ID
    ph_topoetcone40_tightID = r.TH1F("ph_topoetcone40_tightID","ph_topoetcone40_tightID;(topoetcone40-2.45 GeV)/p_{T}^{#gamma}; Entries/0.05", 200, -0.5, 1.5)
    ph_ptcone20_tightID = r.TH1F("ph_ptcone20_tightID","ph_ptcone20_tightID;(ptcone20)/p_{T}^{#gamma}; Entries/0.01", 100, 0.0, 1.0)

    ph_topoetcone40_mediumID = r.TH1F("ph_topoetcone40_mediumID","ph_topoetcone40_mediumID;(topoetcone40-2.45 GeV)/p_{T}^{#gamma}; Entries/0.05", 200, -0.5, 1.5)
    ph_ptcone20_mediumID = r.TH1F("ph_ptcone20_mediumID","ph_ptcone20_mediumID;(ptcone20)/p_{T}^{#gamma}; Entries/0.01", 100, 0.0, 1.0)


    # 2D plots
    ph_topoetcone40_pt_tightID = r.TH2F("ph_topoetcone40_pt_tightID",
                                            "ph_topoetcone40_pt_tightID;p_{T}^{#gamma} [GeV];(topoetcone40-2.45 GeV)/p_{T}^{#gamma}",
                                            10, 0, 100,
                                            20, -0.5, 1.5)
    
    ph_topoetcone20_pt_tightID = r.TH2F("ph_topoetcone20_pt_tightID",
                                            "ph_topoetcone20_pt_tightID;p_{T}^{#gamma} [GeV];(topoetcone20)/p_{T}^{#gamma}",
                                            10, 0, 100,
                                            20, -0.25, 0.75)
    
    ph_topoetcone40_pt_mediumID = r.TH2F("ph_topoetcone40_pt_mediumID",
                                            "ph_topoetcone40_pt_mediumID;p_{T}^{#gamma} [GeV];(topoetcone40-2.45 GeV)/p_{T}^{#gamma}",
                                            10, 0, 100,
                                            20, -0.5, 1.5)
    
    ph_topoetcone20_pt_mediumID = r.TH2F("ph_topoetcone20_pt_mediumID",
                                            "ph_topoetcone20_pt_mediumID;p_{T}^{#gamma} [GeV];(topoetcone20)/p_{T}^{#gamma}",
                                            10, 0, 100,
                                            20, -0.25, 0.75)
    
    hists=[ph_pt_truth,
           ph_pt_reco,
           ph_pt_baseline,
           ph_pt_tightID,
           ph_pt_mediumID,
           ph_pt_tightIso,
           ph_pt_tightID_tightIso,
           ph_pt_mediumID_tightIso,
           ph_pt_looseIso,
           ph_pt_tightID_looseIso,
           ph_pt_mediumID_looseIso,
           ph_topoetcone40_tightID,
           ph_ptcone20_tightID,
           ph_topoetcone40_pt_tightID,
           ph_topoetcone20_pt_tightID,
           ph_topoetcone40_mediumID,
           ph_ptcone20_mediumID,
           ph_topoetcone40_pt_mediumID,
           ph_topoetcone20_pt_mediumID,
           ]

    for i in hists:
        i.Sumw2()

    eventcount=0
    totalevents=t.GetEntriesFast()
    for e in t:
        eventcount+=1
        if eventcount % 10000 == 0:
            print(f"Processed {eventcount:7d} / {totalevents} events")

        truthph_index=-1
        truthph_tlv=None
        for i in range(len(e.truthph_pt)):
            if e.truthph_pt[i] < 10000:
                continue
            if abs(e.truthph_eta[i])>2.37 or (abs(e.truthph_eta[i])>1.37 and abs(e.truthph_eta[i])<1.52):
                continue
            
            if e.truthph_origin[i] != 22:
                continue
            
            # isolation? overlap?

            ph_pt_truth.Fill(e.truthph_pt[i]/1000.)
            truthph_index=i
            truthph_tlv=r.TLorentzVector()
            truthph_tlv.SetPtEtaPhiM(e.truthph_pt[i],e.truthph_eta[i],e.truthph_phi[i],0)
            break

        if truthph_tlv is None:
            continue

        i_baseline=0
        for i in range(len(e.ph_pt_NOSYS)):
            if e.ph_pt_NOSYS[i]<7000:
                continue
            if abs(e.ph_eta[i])>2.37 or (abs(e.ph_eta[i])>1.37 and abs(e.ph_eta[i])<1.52):
                continue

            ph_tlv = r.TLorentzVector()
            ph_tlv.SetPtEtaPhiM(e.ph_truthpt[i],e.ph_trutheta[i],e.ph_truthphi[i],0)
            if ph_tlv.DeltaR(truthph_tlv)>0.02:
                continue

            truthpt=e.ph_truthpt[i]
            recopt=e.ph_pt_NOSYS[i]

            ph_pt_reco.Fill(truthpt/1000.)

            if not (ord(e.ph_select_baseline_NOSYS[i])>0):
                continue

            #print(e.eventNumber)
            #print(i)
            
            sf_baseline = e.ph_id_effSF_baseline_NOSYS[i]
            sf_tightID  = e.ph_id_effSF_tightID_NOSYS[i]
            sf_tightIso = e.ph_id_effSF_tightIso_NOSYS[i]
            sf_mediumID = e.ph_id_effSF_mediumID_NOSYS[i]
            sf_looseIso = e.ph_id_effSF_looseIso_NOSYS[i]

            tightID  = (ord(e.ph_select_tightID_NOSYS[i] ) > 0)
            tightIso = (ord(e.ph_select_tightIso_NOSYS[i]) > 0)
            mediumID = (ord(e.ph_select_mediumID_NOSYS[i]) > 0)
            looseIso = (ord(e.ph_select_looseIso_NOSYS[i]) > 0)

            topoetcone20 = e.ph_topoetcone20_NOSYS[i_baseline]
            topoetcone40 = e.ph_topoetcone40_NOSYS[i_baseline]
            ptcone20     = e.ph_ptcone20_NOSYS[i_baseline]
            
            ph_pt_baseline.Fill(truthpt/1000., sf_baseline)

            if tightID:
                ph_pt_tightID               .Fill(truthpt/1000.                             , sf_tightID)
                ph_topoetcone40_tightID     .Fill((topoetcone40-2450.)/recopt               , sf_tightID)
                ph_ptcone20_tightID         .Fill(ptcone20/recopt                           , sf_tightID)
                ph_topoetcone40_pt_tightID  .Fill(truthpt/1000.,(topoetcone40-2450.)/recopt , sf_tightID)
                ph_topoetcone20_pt_tightID  .Fill(truthpt/1000.,topoetcone20/recopt         , sf_tightID)
                
            if mediumID:
                ph_pt_mediumID              .Fill(truthpt/1000.                             , sf_mediumID)
                ph_topoetcone40_mediumID    .Fill((topoetcone40-2450.)/recopt               , sf_mediumID)
                ph_ptcone20_mediumID        .Fill(ptcone20/recopt                           , sf_mediumID)
                ph_topoetcone40_pt_mediumID .Fill(truthpt/1000.,(topoetcone40-2450.)/recopt , sf_mediumID)
                ph_topoetcone20_pt_mediumID .Fill(truthpt/1000.,topoetcone20/recopt         , sf_mediumID)
                
            if tightIso:
                ph_pt_tightIso              .Fill(truthpt/1000., sf_tightIso)
                
            if looseIso:
                ph_pt_looseIso              .Fill(truthpt/1000., sf_looseIso)
                
            if tightID and tightIso:
                ph_pt_tightID_tightIso      .Fill(truthpt/1000., sf_tightID*sf_tightIso)
                
            if tightID and looseIso:
                ph_pt_tightID_looseIso      .Fill(truthpt/1000., sf_tightID*sf_looseIso)
            
            if mediumID and tightIso:
                ph_pt_mediumID_tightIso     .Fill(truthpt/1000., sf_mediumID*sf_tightIso)
                
            if mediumID and looseIso:
                ph_pt_mediumID_looseIso     .Fill(truthpt/1000., sf_mediumID*sf_looseIso)

            i_baseline+=1
            break


    for i in hists:
        i.Write()
    
    output.Close()



#tag="545767.N2_200_N1_190_WB.mc20d"

ntupledirs=glob.glob("/data/mhance/SUSY/ntuples/v3_6/user.bhodkins.RadiativeDecays.*.v3.*__NOFILTER_ANALYSIS.root")

dsids=[]
for i in ntupledirs:
    tag=i.replace("/data/mhance/SUSY/ntuples/v3_6/user.bhodkins.RadiativeDecays.","").replace(".v3.3","").replace(".v3.6","").replace("__NOFILTER_ANALYSIS.root","")
    print(tag)
    dsid=tag.split(".")[0]
    physicsshort=tag.split(".")[1]
    phystag=dsid+"."+physicsshort
    if phystag not in dsids:
        dsids.append(phystag)
    processfile(tag)

for dsid in dsids:
    os.system(f"hadd -f efficoutputs/{dsid}.root efficoutputs/{dsid}.*.root")
