import ROOT as r
import glob


colors={"reco": r.kBlack,
        "baseline": r.kRed,
        "tightID": r.kRed+1,
        "mediumID": r.kRed-7,
        "tightIso": r.kRed+2,
        "looseIso": r.kRed+4,
        "tightCOIso": r.kOrange,
        "hybridIso": r.kOrange+1,
        "hybridCOIso": r.kOrange+2,
        "tightID_looseIso": r.kViolet,
        "tightID_tightIso": r.kViolet-1,
        "tightID_tightCOIso": r.kViolet-2,
        "tightID_hybridIso": r.kViolet-3,
        "tightID_hybridCOIso": r.kViolet-4,
        "mediumID_tightIso": r.kRed-6,
        "mediumID_looseIso": r.kRed-5,
        "truth": r.kBlue,
        "truth_SUSY_fiducial": r.kBlue+1,
        "truth_SUSY_all": r.kBlue+2
        }

markers={"reco": 8,
         "baseline": 2,
         "tightID": 3,
         "mediumID": 29,
         "tightIso": 5,
         "tightCOIso": 36,
         "looseIso": 33,
         "hybridIso": 37,
         "hybridCOIso": 38,
         "tightID_looseIso": 4,
         "tightID_tightIso": 28,
         "tightID_tightCOIso": 39,
         "tightID_hybridIso": 40,
         "tightID_hybridCOIso": 41,
         "mediumID_tightIso": 30,
         "mediumID_looseIso": 32,
         "truth": 34,
         "truth_SUSY_fiducial": 35
        }

def setStyles(objs):
    for tag,obj in objs.items():
        obj.SetLineColor(colors[tag])
        obj.SetFillColor(0)
        obj.SetMarkerColor(colors[tag])
        obj.SetMarkerStyle(markers[tag])
        obj.SetTitle(";True p_{T}^{#gamma} [GeV];Efficiency")
        obj.Write(f"eff_{tag}")


def makeplots(f,denom="truth",isoOnly=False):
    rf = r.TFile(f,"RO")
    h={}

    std_plots=[
        "truth_SUSY_all",
        "truth_SUSY_fiducial",
        "truth",
        "reco",
        "baseline"
    ]
    
    iso_plots=[
        "looseIso",
        "tightIso",
        "tightCOIso",
        "hybridIso",
        "hybridCOIso"
    ]

    ID_plots=[
        "baseline",
        "mediumID",
        "tightID"
    ]

    IDIso_tightID_plots=[
        "tightID_looseIso", 
        "tightID_tightIso",
        "tightID_tightCOIso",
        "tightID_hybridIso",
        "tightID_hybridCOIso"
    ]

    IDIso_mediumID_plots=[
        "mediumID_looseIso", 
        "mediumID_tightIso",
        "mediumID_tightCOIso",
        "mediumID_hybridIso",
        "mediumID_hybridCOIso"
    ]
    
    IDiso_plots=IDIso_tightID_plots+IDIso_mediumID_plots

    all_plots= std_plots + iso_plots + ID_plots + IDiso_plots

    plots=None
    if isoOnly:
        #plots=std_plots+iso_plots
        plots=std_plots+["tightID"]+IDIso_tightID_plots
    elif denom=="truth_SUSY_all":
        plots=std_plots
    else:
        plots=all_plots
    #print(plots)
    
    for t in plots:
        if denom=="baseline" and (t=="truth" or t=="reco"):
            continue
        if denom=="truth" and ("truth_SUSY" in t):
            continue
        h[t] = rf.Get(f"ph_pt_{t}")

    ro = r.TFile(f"{f[:-5]}_effs.root","RECREATE")   

    h_denom=h[denom]
    
    effs={}
    if "truth_SUSY_fiducial" in h and not isoOnly:
        effs["truth_SUSY_fiducial"] = r.TEfficiency(h["truth_SUSY_fiducial"], h_denom)
    if "truth" in h:
        effs["truth"] = r.TEfficiency(h["truth"], h_denom)
    if "reco" in h:
        effs["reco"] = r.TEfficiency(h["reco"],h_denom)
    if "baseline" in h:
        effs["baseline"] = r.TEfficiency(h["baseline"],h_denom)
        
    for Iso in iso_plots:
        if Iso in h:
            effs[Iso] = r.TEfficiency(h[Iso],h_denom)

    for ID in ["mediumID", "tightID"]:
        if ID in h:
            effs[ID] = r.TEfficiency(h[ID],h_denom)

        for Iso in iso_plots:
            if ID+"_"+Iso in h:
                effs[ID+"_"+Iso] = r.TEfficiency(h[ID+"_"+Iso],h_denom)

    setStyles(effs)
    
    c=r.TCanvas("effics","effics",800,600)
    c.SetGridy(1)
    c.SetLeftMargin(0.15)
    c.SetBottomMargin(0.15)
    drawhist=""
    if denom=="truth":
        drawhist="reco"
    elif denom=="truth_SUSY_all":
        drawhist="truth_SUSY_fiducial"
    elif denom=="baseline" and "looseIso" in effs:
        drawhist="looseIso"
    elif denom=="baseline":
        drawhist="tightID_looseIso"
    #print(list(effs.keys()))
        
    effs[drawhist].Draw()
    leg=r.TLegend(0.5,0.2,0.85,0.3+0.035*(len(effs)-2))
    leg.AddEntry(effs[drawhist], drawhist)
    for tag,obj in effs.items():
        if tag==drawhist: continue
        if tag==denom: continue
        #if "_" not in tag: continue
        obj.Draw("same")
        if tag == "baseline":
            leg.AddEntry(obj, "looseID")
        elif denom=="truth":
            leg.AddEntry(obj, f"looseID+{tag.replace('_','+')}")
        elif denom=="baseline":
            leg.AddEntry(obj, f"{tag.replace('_','+')}")
        elif denom=="truth_SUSY_all":
            leg.AddEntry(obj, f"{tag.replace('_','+')}")
            
    leg.Draw()
    c.Update()
    
    axishist=effs[drawhist].GetPaintedGraph().GetHistogram()
    axishist.GetYaxis().SetTitle("Efficiency")
    axishist.GetYaxis().SetRangeUser(0,1)
    axishist.GetYaxis().SetLabelSize(0.05)
    axishist.GetXaxis().SetLabelSize(0.05)
    axishist.GetXaxis().SetTitleSize(0.06)
    axishist.GetYaxis().SetTitleSize(0.06)
    axishist.GetXaxis().SetTitleOffset(1.1)
    
    c.Update()
    r.gPad.Update()


    tl=r.TLatex()
    tl.SetNDC()
    tl.SetTextSize(0.03)
    tl.SetTextFont(82)
    tl.DrawLatex(0.56, 0.92,f"{f[13:-5]}")
    
    c.Print((f"{f[:-5]}_effs_{denom}.pdf").replace("efficoutputs/","efficoutputs/pdfs/"))


    # pt plot
    if False:
        c2=r.TCanvas("pt","pt",800,600)
        c2.SetLogy(1)
        
        #for t in ["truth", "baseline", "tightID", "tightIso", "looseIso", "tightID_tightIso", "tightID_looseIso"]:
        h["truth"].SetLineColor(r.kBlue)
        h["truth"].Draw()
        h["baseline"].SetLineColor(r.kRed)
        h["baseline"].Draw("same")
        h["tightID"].SetLineColor(r.kRed+1)
        h["tightID"].Draw("same")
        h["tightIso"].SetLineColor(r.kRed+2)
        h["tightIso"].Draw("same")
        h["tightID_tightIso"].SetLineColor(r.kRed+3)
        h["tightID_tightIso"].Draw("same")
        
        c2.Print(f"{f[:-5]}_pt.pdf")
        c2.Write()
    
    ro.Close()
    rf.Close()

gridpoints=[]
gridpoints += glob.glob("efficoutputs/*WB.mc20.root")
gridpoints += glob.glob("efficoutputs/*WB.mc23.root")

for f in gridpoints:
    print(f)
    makeplots(f,"baseline",True)
    makeplots(f,"truth",True)
    #makeplots(f,"truth",False)
    #makeplots(f,"truth_SUSY_all",False)
