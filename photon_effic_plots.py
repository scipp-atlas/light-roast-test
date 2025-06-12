import ROOT as r
import glob

colors={"reco": r.kRed,
        "tightID": r.kRed+1,
        "mediumID": r.kRed-7,
        "tightIso": r.kRed+2,
        "looseIso": r.kRed+4,
        "tightID_tightIso": r.kRed+3,
        "tightID_looseIso": r.kRed-8,
        "mediumID_tightIso": r.kRed-6,
        "mediumID_looseIso": r.kRed-5,
        }

markers={"reco": 2,
         "tightID": 3,
         "mediumID": 29,
         "tightIso": 5,
         "looseIso": 33,
         "tightID_tightIso": 4,
         "tightID_looseIso": 28,
         "mediumID_tightIso": 30,
         "mediumID_looseIso": 32,
        }

def setStyles(objs):
    for tag,obj in objs.items():
        obj.SetLineColor(colors[tag])
        obj.SetFillColor(0)
        obj.SetMarkerColor(colors[tag])
        obj.SetMarkerStyle(markers[tag])
        obj.Write(f"eff_{tag}")

def makeplots(f):
    rf = r.TFile(f,"RO")
    h={}
    for t in ["truth", "reco",
              "tightID", "tightIso",
              "mediumID", "looseIso",
              "tightID_tightIso", "tightID_looseIso",
              "mediumID_tightIso", "mediumID_looseIso",
              ]:
        h[t] = rf.Get(f"ph_pt_{t}")

    ro = r.TFile(f"{f[:-5]}_effs.root","RECREATE")   

    effs={}
    effs["reco"] = r.TEfficiency(h["reco"],h["truth"])
    for Iso in ["looseIso", "tightIso"]:
        effs[Iso] = r.TEfficiency(h[Iso],h["truth"])

    for ID in ["mediumID", "tightID"]:
        effs[ID] = r.TEfficiency(h[ID],h["truth"])

        for Iso in ["looseIso", "tightIso"]:
            effs[ID+"_"+Iso] = r.TEfficiency(h[ID+"_"+Iso],h["truth"])

    setStyles(effs)
    
    effs["reco"].SetTitle(";True p_{T}^{#gamma} [GeV];Efficiency")
    
    c=r.TCanvas("effics","effics",800,600)
    c.SetGridy(1)
    c.SetLeftMargin(0.15)
    c.SetBottomMargin(0.15)
    effs["reco"]            .Draw()
    leg=r.TLegend(0.3,0.2,0.85,0.5)
    leg.AddEntry(effs["reco"], "looseID")
    for tag,obj in effs.items():
        if tag=="reco": continue
        if "_" not in tag: continue
        obj.Draw("same")
        leg.AddEntry(obj, f"looseID+{tag.replace('_','+')}")
    leg.Draw()
    c.Update()
    
    axishist=effs["reco"].GetPaintedGraph().GetHistogram()
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
    tl.DrawLatex(0.6, 0.92,f"{f[13:-5]}")
    
    c.Print(f"{f[:-5]}_effs.pdf")


    # pt plot
    if False:
        c2=r.TCanvas("pt","pt",800,600)
        c2.SetLogy(1)
        
        #for t in ["truth", "reco", "tightID", "tightIso", "looseIso", "tightID_tightIso", "tightID_looseIso"]:
        h["truth"].SetLineColor(r.kBlue)
        h["truth"].Draw()
        h["reco"].SetLineColor(r.kRed)
        h["reco"].Draw("same")
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


gridpoints=glob.glob("efficoutputs/*WB.root")

for f in gridpoints:
    print(f)
    makeplots(f)

