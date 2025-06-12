import ROOT as r
import json
from pathlib import Path
import glob
import os

def processfile(filetag):
    filename=glob.glob(f"/data/mhance/SUSY/ntuples/v3_3/user.bhodkins.RadiativeDecays.{filetag}.v3.3__NOFILTER_ANALYSIS.root/user.bhodkins.*.ANALYSIS.root")
    
    f=r.TFile(filename[0],"RO")
    t=f.Get("analysis")
    
    output = r.TFile(f"efficoutputs/{filetag}.root","RECREATE")

    photon_pt_truth = r.TH1F("photon_pt_truth", "photon_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    photon_pt_reco = r.TH1F("photon_pt_reco", "photon_pT;truth #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    photon_pt_tightID = r.TH1F("photon_pt_tightID", "photon_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    photon_pt_tightIso = r.TH1F("photon_pt_tightIso", "photon_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    photon_pt_looseIso = r.TH1F("photon_pt_looseIso", "photon_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    photon_pt_tightID_tightIso = r.TH1F("photon_pt_tightID_tightIso", "photon_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)
    photon_pt_tightID_looseIso = r.TH1F("photon_pt_tightID_looseIso", "photon_pT;true #gamma p_{T} [GeV]; Entries/(5 GeV)", 20, 0, 100)

    photon_topoetcone40_tightID = r.TH1F("ph_topoetcone40_tightID","ph_topoetcone40_tightID;(topoetcone40-2.45 GeV)/p_{T}^{#gamma}; Entries/0.05", 200, -0.5, 1.5)
    photon_ptcone20_tightID = r.TH1F("ph_ptcone20_tightID","ph_ptcone20_tightID;(ptcone20)/p_{T}^{#gamma}; Entries/0.01", 100, 0.0, 1.0)

    photon_topoetcone40_pt_tightID = r.TH2F("ph_topoetcone40_pt_tightID",
                                            "ph_topoetcone40_pt_tightID;p_{T}^{#gamma} [GeV];(topoetcone40-2.45 GeV)/p_{T}^{#gamma}",
                                            10, 0, 100,
                                            20, -0.5, 1.5)
    
    photon_topoetcone20_pt_tightID = r.TH2F("ph_topoetcone20_pt_tightID",
                                            "ph_topoetcone20_pt_tightID;p_{T}^{#gamma} [GeV];(topoetcone20)/p_{T}^{#gamma}",
                                            10, 0, 100,
                                            20, -0.25, 0.75)
    
    photon_pt_truth.Sumw2()
    photon_pt_reco.Sumw2()
    photon_pt_tightID.Sumw2()
    photon_pt_tightIso.Sumw2()
    photon_pt_tightID_tightIso.Sumw2()
    
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

            photon_pt_truth.Fill(e.truthph_pt[i]/1000.)
            truthph_index=i
            truthph_tlv=r.TLorentzVector()
            truthph_tlv.SetPtEtaPhiM(e.truthph_pt[i],e.truthph_eta[i],e.truthph_phi[i],0)
            break

        if truthph_tlv is None:
            continue
        
        for i in range(len(e.ph_pt_NOSYS)):
            if e.ph_pt_NOSYS[i]<7000:
                continue
            ph_tlv = r.TLorentzVector()
            ph_tlv.SetPtEtaPhiM(e.ph_truthpt[i],e.ph_trutheta[i],e.ph_truthphi[i],0)
            if ph_tlv.DeltaR(truthph_tlv)>0.02: continue

            pt=e.ph_truthpt[i]/1000.
            sf_id = e.ph_id_effSF_tightID_NOSYS[i]
            sf_iso = e.ph_id_effSF_tightIso_NOSYS[i]
            
            photon_pt_reco.Fill(pt)

            if ord(e.ph_select_tightID_NOSYS[i])>0:
                photon_pt_tightID.Fill(pt, sf_id)
                photon_topoetcone40_tightID.Fill((e.ph_topoetcone40_NOSYS[i]-2450.)/e.ph_pt_NOSYS[i], sf_id)
                photon_ptcone20_tightID.Fill(e.ph_ptcone20_NOSYS[i]/e.ph_pt_NOSYS[i], sf_id)
                photon_topoetcone40_pt_tightID.Fill(e.ph_pt_NOSYS[i]/1000.,(e.ph_topoetcone40_NOSYS[i]-2450.)/e.ph_pt_NOSYS[i],sf_id)
                photon_topoetcone20_pt_tightID.Fill(e.ph_pt_NOSYS[i]/1000.,(e.ph_topoetcone20_NOSYS[i])/e.ph_pt_NOSYS[i],sf_id)
            if ord(e.ph_select_tightIso_NOSYS[i])>0:
                photon_pt_tightIso.Fill(pt, sf_iso)
            if ord(e.ph_select_looseIso_NOSYS[i])>0:
                photon_pt_looseIso.Fill(pt, e.ph_id_effSF_looseIso_NOSYS[i])
            if ord(e.ph_select_tightID_NOSYS[i])>0 and ord(e.ph_select_tightIso_NOSYS[i])>0:
                photon_pt_tightID_tightIso.Fill(pt, sf_id*sf_iso)
            if ord(e.ph_select_tightID_NOSYS[i])>0 and ord(e.ph_select_looseIso_NOSYS[i])>0:
                photon_pt_tightID_looseIso.Fill(pt, sf_id*e.ph_id_effSF_looseIso_NOSYS[i])
            
            break
            # overlap removal?

    hists=[photon_pt_truth,
           photon_pt_reco,
           photon_pt_tightID,
           photon_pt_tightIso,
           photon_pt_tightID_tightIso,
           photon_pt_looseIso,
           photon_pt_tightID_looseIso,
           photon_topoetcone40_tightID,
           photon_ptcone20_tightID,
           photon_topoetcone40_pt_tightID,
           photon_topoetcone20_pt_tightID,
           ]

    for i in hists:
        i.Write()
    
    output.Close()



#tag="545767.N2_200_N1_190_WB.mc20d"

ntupledirs=glob.glob("/data/mhance/SUSY/ntuples/v3_3/user.bhodkins.RadiativeDecays.*.v3.3__NOFILTER_ANALYSIS.root")

dsids=[]
for i in ntupledirs:
    tag=i.replace("/data/mhance/SUSY/ntuples/v3_3/user.bhodkins.RadiativeDecays.","").replace(".v3.3__NOFILTER_ANALYSIS.root","")
    print(tag)
    dsid=tag.split(".")[0]
    physicsshort=tag.split(".")[1]
    phystag=dsid+"."+physicsshort
    if phystag not in dsids:
        dsids.append(phystag)
    processfile(tag)

for dsid in dsids:
    os.system(f"hadd -f efficoutputs/{dsid}.root efficoutputs/{dsid}.*.root")
