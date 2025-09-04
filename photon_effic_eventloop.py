import ROOT as r
import json
from pathlib import Path
import glob
import os

run2path="/data/mhance/SUSY/ntuples/v3_6"
run3path="/data/mhance/SUSY/ntuples/v3.18"

runpath=run2path

def processfile(filetag):
    filename=glob.glob(f"{runpath}/user.bhodkins.RadiativeDecays.{filetag}.v3.*__NOFILTER_ANALYSIS.root/user.bhodkins.*.ANALYSIS.root")
    
    f=r.TFile(filename[0],"RO")
    t=f.Get("analysis")
    
    output = r.TFile(f"efficoutputs/{filetag}.root","RECREATE")

    # 1D plots of pT
    ph_pt_truth_SUSY_all = r.TH1F("ph_pt_truth_SUSY_all", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_truth_SUSY_fiducial = r.TH1F("ph_pt_truth_SUSY_fiducial", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_truth = r.TH1F("ph_pt_truth", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_reco = r.TH1F("ph_pt_reco", "ph_pT;truth #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_baseline = r.TH1F("ph_pt_baseline", "ph_pT;truth #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_tightID = r.TH1F("ph_pt_tightID", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID = r.TH1F("ph_pt_mediumID", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_tightIso = r.TH1F("ph_pt_tightIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_looseIso = r.TH1F("ph_pt_looseIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_hybridIso = r.TH1F("ph_pt_hybridIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_tightCOIso = r.TH1F("ph_pt_tightCOIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_hybridCOIso = r.TH1F("ph_pt_hybridCOIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_tightID_tightIso = r.TH1F("ph_pt_tightID_tightIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_tightID_looseIso = r.TH1F("ph_pt_tightID_looseIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_tightID_hybridIso = r.TH1F("ph_pt_tightID_hybridIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_tightID_tightCOIso = r.TH1F("ph_pt_tightID_tightCOIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_tightID_hybridCOIso = r.TH1F("ph_pt_tightID_hybridCOIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    ph_pt_mediumID_tightIso = r.TH1F("ph_pt_mediumID_tightIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID_looseIso = r.TH1F("ph_pt_mediumID_looseIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID_hybridIso = r.TH1F("ph_pt_mediumID_hybridIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID_tightCOIso = r.TH1F("ph_pt_mediumID_tightCOIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    ph_pt_mediumID_hybridCOIso = r.TH1F("ph_pt_mediumID_hybridCOIso", "ph_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

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
    
    hists=[ph_pt_truth_SUSY_all,
           ph_pt_truth_SUSY_fiducial,
           ph_pt_truth,
           ph_pt_reco,
           ph_pt_baseline,
           ph_pt_tightID,
           ph_pt_mediumID,
           ph_pt_tightIso,
           ph_pt_tightID_tightIso,
           ph_pt_mediumID_tightIso,
           ph_pt_tightCOIso,
           ph_pt_tightID_tightCOIso,
           ph_pt_mediumID_tightCOIso,
           ph_pt_looseIso,
           ph_pt_tightID_looseIso,
           ph_pt_mediumID_looseIso,
           ph_pt_hybridIso,
           ph_pt_tightID_hybridIso,
           ph_pt_mediumID_hybridIso,
           ph_pt_hybridCOIso,
           ph_pt_tightID_hybridCOIso,
           ph_pt_mediumID_hybridCOIso,
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

        truthjet_tlvs=[]
        for j in range(len(e.truthjet_pt)):
            truthjet_tlv = r.TLorentzVector()
            truthjet_tlv.SetPtEtaPhiM(e.truthjet_pt[j],e.truthjet_eta[j],e.truthjet_phi[j],0)
            truthjet_tlvs.append(truthjet_tlv)

        truthel_tlvs=[]
        for el in range(len(e.truthel_pt)):
            truthel_tlv = r.TLorentzVector()
            truthel_tlv.SetPtEtaPhiM(e.truthel_pt[el],e.truthel_eta[el],e.truthel_phi[el],0)
            truthel_tlvs.append(truthel_tlv)
        
        truthmu_tlvs=[]
        for mu in range(len(e.truthmu_pt)):
            truthmu_tlv = r.TLorentzVector()
            truthmu_tlv.SetPtEtaPhiM(e.truthmu_pt[mu],e.truthmu_eta[mu],e.truthmu_phi[mu],0)
            truthmu_tlvs.append(truthmu_tlv)

        truthph_index=-1
        truthph_tlv=None
        for i in range(len(e.truthph_pt)):

            # SUSY photon requirement
            if e.truthph_origin[i] != 22:
                continue

            truthph_cand_tlv = r.TLorentzVector()
            truthph_cand_tlv.SetPtEtaPhiM(e.truthph_pt[i],e.truthph_eta[i],e.truthph_phi[i],0)

            ph_pt_truth_SUSY_all.Fill(truthph_cand_tlv.Pt()/1000.)
            
            # Fiducial cuts
            if e.truthph_pt[i] < 10000:
                continue
                
            if abs(e.truthph_eta[i])>2.37 or (abs(e.truthph_eta[i])>1.37 and abs(e.truthph_eta[i])<1.52):
                continue
            
            ph_pt_truth_SUSY_fiducial.Fill(truthph_cand_tlv.Pt()/1000.)
            
            # ==================================================================================
            # isolation/dR vetos
            mindR_truthph_truthjet=999.
            for truthjet_tlv in truthjet_tlvs:
                dR = truthph_cand_tlv.DeltaR(truthjet_tlv)
                if dR<0.10: continue
                mindR_truthph_truthjet = min(mindR_truthph_truthjet,dR)
            if mindR_truthph_truthjet < 0.4:
                # debug if desired
                #for truthjet_tlv in truthjet_tlvs:
                #    dR = truthph_cand_tlv.DeltaR(truthjet_tlv)
                #    print(f"ph_pt: {truthph_cand_tlv.Pt()/1000.:5.0f}; ph_eta: {truthph_cand_tlv.Eta():5.2f}; ph_phi: {truthph_cand_tlv.Phi():5.2f}; jet_pt: {truthjet_tlv.Pt()/1000.:5.0f}; jet_eta: {truthjet_tlv.Eta():5.2f}; jet_phi: {truthjet_tlv.Phi():5.2f}; dR={dR:5.2f}")
                continue

            mindR_truthph_truthel=999.
            for truthel_tlv in truthel_tlvs:
                dR = truthph_cand_tlv.DeltaR(truthel_tlv)
                if dR<0.05: continue
                mindR_truthph_truthel = min(mindR_truthph_truthel,dR)
            if mindR_truthph_truthel < 0.4: 
                continue

            mindR_truthph_truthmu=999.
            for truthmu_tlv in truthmu_tlvs:
                dR = truthph_cand_tlv.DeltaR(truthmu_tlv)
                mindR_truthph_truthmu = min(mindR_truthph_truthmu,dR)
            if mindR_truthph_truthmu < 0.4: 
                continue
            # ==================================================================================

            ph_pt_truth.Fill(truthph_cand_tlv.Pt()/1000.)
            
            truthph_index=i
            truthph_tlv = truthph_cand_tlv
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

            sf_baseline     = 1.
            sf_mediumID     = 1.
            sf_tightID      = 1.
            sf_tightIso     = 1.
            sf_tightCOIso   = 1.
            sf_looseIso     = 1.
            sf_hybridIso    = 1.
            sf_hybridCOIso  = 1.

            run2idsfgood  = False # still have identical values across loose/medium/tight, but otherwise OK.  Removing just to have apples-to-apples comparisons with Run 3
            run3idsfgood  = False # somehow also problematic
            run2isosfgood = False # not good in this round of ntuples, but will be in next round
            run3isosfgood = False # not good at all

            if runpath is run2path:                
                if run2idsfgood:
                    sf_baseline     = e.ph_id_effSF_baseline_NOSYS[i]
                    sf_mediumID     = e.ph_id_effSF_mediumID_NOSYS[i]
                    sf_tightID      = e.ph_id_effSF_tightID_NOSYS[i]
            
                if run2isosfgood:
                    sf_tightIso     = e.ph_isol_effSF_tightIso_NOSYS[i]
                    sf_tightCOIso   = e.ph_isol_effSF_tightCOIso_NOSYS[i]
                    sf_looseIso     = e.ph_isol_effSF_looseIso_NOSYS[i]
                    sf_hybridIso    = e.ph_isol_effSF_looseIso_NOSYS[i] if (recopt>20000.) else e.ph_isol_effSF_tightIso_NOSYS[i]
                    sf_hybridCOIso  = e.ph_isol_effSF_looseIso_NOSYS[i] if (recopt>25000.) else e.ph_isol_effSF_tightCOIso_NOSYS[i]

            elif runpath is run3path:
                if run3idsfgood:
                    sf_baseline     = e.ph_id_effSF_baseline_NOSYS[i]
                    sf_mediumID     = e.ph_id_effSF_mediumID_NOSYS[i]
                    sf_tightID      = e.ph_id_effSF_tightID_NOSYS[i]
            
                if run3isosfgood:
                    sf_tightIso     = e.ph_isol_effSF_tightIso_NOSYS[i]
                    sf_tightCOIso   = e.ph_isol_effSF_tightCOIso_NOSYS[i]
                    sf_looseIso     = e.ph_isol_effSF_looseIso_NOSYS[i]
                    sf_hybridIso    = e.ph_isol_effSF_looseIso_NOSYS[i] if (recopt>20000.) else e.ph_isol_effSF_tightIso_NOSYS[i]
                    sf_hybridCOIso  = e.ph_isol_effSF_looseIso_NOSYS[i] if (recopt>25000.) else e.ph_isol_effSF_tightCOIso_NOSYS[i]

            
            tightID      = (ord(e.ph_select_tightID_NOSYS[i] ) > 0)
            tightIso     = (ord(e.ph_select_tightIso_NOSYS[i]) > 0)
            tightCOIso   = (ord(e.ph_select_tightCOIso_NOSYS[i]) > 0)
            mediumID     = (ord(e.ph_select_mediumID_NOSYS[i]) > 0)
            looseIso     = (ord(e.ph_select_looseIso_NOSYS[i]) > 0)
            hybridIso    = (ord(e.ph_select_looseIso_NOSYS[i]) > 0) if (recopt>20000.) else (ord(e.ph_select_tightIso_NOSYS[i]) > 0)
            hybridCOIso  = (ord(e.ph_select_looseIso_NOSYS[i]) > 0) if (recopt>25000.) else (ord(e.ph_select_tightCOIso_NOSYS[i]) > 0)

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
                
            if tightCOIso:
                ph_pt_tightCOIso            .Fill(truthpt/1000., sf_tightCOIso)
                
            if looseIso:
                ph_pt_looseIso              .Fill(truthpt/1000., sf_looseIso)

            if hybridIso:
                ph_pt_hybridIso             .Fill(truthpt/1000., sf_hybridIso)
                
            if hybridCOIso:
                ph_pt_hybridCOIso           .Fill(truthpt/1000., sf_hybridCOIso)
                
            if tightID and tightIso:
                ph_pt_tightID_tightIso      .Fill(truthpt/1000., sf_tightID*sf_tightIso)
                
            if tightID and tightCOIso:
                ph_pt_tightID_tightCOIso    .Fill(truthpt/1000., sf_tightID*sf_tightCOIso)
                
            if tightID and looseIso:
                ph_pt_tightID_looseIso      .Fill(truthpt/1000., sf_tightID*sf_looseIso)
            
            if tightID and hybridIso:
                ph_pt_tightID_hybridIso     .Fill(truthpt/1000., sf_tightID*sf_hybridIso)
            
            if tightID and hybridCOIso:
                ph_pt_tightID_hybridCOIso   .Fill(truthpt/1000., sf_tightID*sf_hybridCOIso)
            
            if mediumID and tightIso:
                ph_pt_mediumID_tightIso     .Fill(truthpt/1000., sf_mediumID*sf_tightIso)
                
            if mediumID and tightCOIso:
                ph_pt_mediumID_tightCOIso   .Fill(truthpt/1000., sf_mediumID*sf_tightCOIso)
                
            if mediumID and looseIso:
                ph_pt_mediumID_looseIso     .Fill(truthpt/1000., sf_mediumID*sf_looseIso)

            if mediumID and hybridIso:
                ph_pt_mediumID_hybridIso    .Fill(truthpt/1000., sf_mediumID*sf_hybridIso)

            if mediumID and hybridCOIso:
                ph_pt_mediumID_hybridCOIso  .Fill(truthpt/1000., sf_mediumID*sf_hybridCOIso)

            i_baseline+=1
            break


    for i in hists:
        i.Write()
    
    output.Close()



#tag="545767.N2_200_N1_190_WB.mc20d"

ntupledirs=glob.glob(f"{runpath}/user.bhodkins.RadiativeDecays.*.v3.*__NOFILTER_ANALYSIS.root")

dsids=[]
for i in ntupledirs:
    tag=i.replace(f"{runpath}/user.bhodkins.RadiativeDecays.","").replace(".v3.3","").replace(".v3.6","").replace(".v3.18","").replace("__NOFILTER_ANALYSIS.root","")
    print(tag)
    dsid=tag.split(".")[0]
    physicsshort=tag.split(".")[1]
    phystag=dsid+"."+physicsshort
    if phystag not in dsids:
        dsids.append(phystag)
    processfile(tag)

for dsid in dsids:
    if runpath==run2path:
        os.system(f"hadd -f efficoutputs/{dsid}.mc20.root efficoutputs/{dsid}.mc20[ade].root")
    else:
        os.system(f"hadd -f efficoutputs/{dsid}.mc23.root efficoutputs/{dsid}.mc23[ad].root")
