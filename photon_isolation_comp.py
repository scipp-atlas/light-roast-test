import ROOT as r

WPDefs={
    "Tight":           0x2ffc00,
    #"LoosePrime2":    0x27fc00,
    #"LoosePrime3":    0x25fc00,
    "LoosePrime4":    0x05fc00,
    #"LoosePrime4a":   0x21fc00,
    #"LoosePrime5":    0x01fc00,
    #"LoosePrimeRun1": 0x45fc01,
}

WPColors={
    "Tight":          r.kBlack,
    "LoosePrime2":    r.kBlue,
    "LoosePrime3":    r.kRed,
    "LoosePrime4":    r.kGreen,
    "LoosePrime4a":   r.kBrown,
    "LoosePrime5":    r.kOrange,
    #"LoosePrimeRun1": r.kYellow+1,
}

f=r.TFile("output_Znunu_CVetoBVeto_mc20.root","RO")
t=f.Get("picontuple")

h_isEM=r.TH1F("isEM","isEM",50,0,50)

nbins={"iso": 25,
       "pt": 100
       }

binlow={"iso": -0.5,
        "pt": 0
        }

binhgh={"iso": 2.0,
        "pt": 100
        }

h={"iso": {},
   "pt": {},
   }

for lp in WPDefs:
    for hist in ["iso","pt"]:
        h[hist][lp] = r.TH1F(f"{hist}_{lp}","{hist}_{lp}",nbins[hist],binlow[hist],binhgh[hist])
        h[hist][lp].SetLineColor(WPColors[lp])
        h[hist][lp].SetMarkerColor(WPColors[lp])

nentries=t.GetEntriesFast()
ecount=0
for e in t:
    ecount+=1
    if (ecount%10000 == 0):
        print(f"Processed {ecount:6d}/{nentries:6d} events")
        
    # event level cuts
    if e.met_met < 200000: continue
    if e.jet_cleanTightBad_prod != 1: continue
    if e.j1_pt < 150000: continue
    if e.ph_pt < 10000: continue
    if e.mindPhiJetMet < 0.4: continue
    if e.nBTagJets > 0: continue
    if e.nElectrons > 0: continue
    if e.nMuons > 0: continue
    if e.nTau20_baseline > 0: continue
    if e.mindPhiGammaJet < 1.5: continue
    if e.ph_select_baseline != 1: continue

    # calculate weight
    weight=1.
    weight *= e.weight_total * e.weight_fjvt_effSF * e.weight_ftag_effSF_GN2v01_Continuous * e.weight_jvt_effSF

    # check which isEM bits are high
    for i in range(50):
        if (e.ph_isEM & (1<<i)) != 0:
            h_isEM.Fill(i,weight)

    iso=(e.ph_topoetcone40-2450.)/e.ph_pt
    pt=e.ph_pt/1000.
            
    # fill some isolation dists
    if (e.ph_truthJFP==1) or (e.ph_truthother==1):
        if (e.ph_select_tightID==1):
            h["iso"]["Tight"].Fill(iso,weight)
            h["pt"]["Tight"].Fill(pt,weight)
        else:
            for lp in WPDefs:
                if lp=="Tight": continue
                if ((e.ph_isEM & WPDefs[lp])==0):
                    h["iso"][lp].Fill(iso,weight)
                    h["pt"][lp].Fill(pt,weight)

c={}

for hist in ["iso","pt"]:
    c[hist]=r.TCanvas(f"c_{hist}",f"c_{hist}",800,600)
    c[hist].SetLogy(1)
    c[hist].SetGridx(1)
    h[hist]["Tight"].Draw()
    for lp in WPDefs:
        if lp=="Tight": continue
        h[hist][lp].Draw("same")

#c2=r.TCanvas("c2","c2",800,600)
#c2.SetLogy(1)
#h_isEM.Draw()
