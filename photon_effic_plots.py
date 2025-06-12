import ROOT as r
import glob

def makeplots(f):
    rf = r.TFile(f,"RO")
    h={}
    for t in ["truth", "reco", "tightID", "tightIso", "looseIso", "tightID_tightIso", "tightID_looseIso"]:
        h[t] = rf.Get(f"photon_pt_{t}")

    ro = r.TFile(f"{f[:-5]}_effs.root","RECREATE")   

    eff_reco   = r.TEfficiency(h["reco"],h["truth"])
    eff_ID     = r.TEfficiency(h["tightID"],h["reco"])
    eff_tIso    = r.TEfficiency(h["tightIso"],h["reco"])
    eff_lIso    = r.TEfficiency(h["looseIso"],h["reco"])
    eff_IDtIso  = r.TEfficiency(h["tightID_tightIso"],h["reco"])
    eff_IDlIso  = r.TEfficiency(h["tightID_looseIso"],h["reco"])
    eff_recoID     = r.TEfficiency(h["tightID"],h["truth"])
    eff_recotIso    = r.TEfficiency(h["tightIso"],h["truth"])
    eff_recolIso    = r.TEfficiency(h["looseIso"],h["truth"])
    eff_recoIDtIso  = r.TEfficiency(h["tightID_tightIso"],h["truth"])
    eff_recoIDlIso  = r.TEfficiency(h["tightID_looseIso"],h["truth"])

    eff_reco.SetLineColor(r.kRed)
    eff_recoID.SetLineColor(r.kRed+1)
    eff_recotIso.SetLineColor(r.kRed+2)
    eff_recolIso.SetLineColor(r.kRed+4)
    eff_recoIDtIso.SetLineColor(r.kRed+3)
    eff_recoIDlIso.SetLineColor(r.kRed-8)
    eff_reco.SetMarkerColor(r.kRed)
    eff_recoID.SetMarkerColor(r.kRed+1)
    eff_recotIso.SetMarkerColor(r.kRed+2)
    eff_recolIso.SetMarkerColor(r.kRed+4)
    eff_recoIDtIso.SetMarkerColor(r.kRed+3)
    eff_recoIDlIso.SetMarkerColor(r.kRed-8)
    eff_reco.SetFillColor(0)
    eff_recoID.SetFillColor(0)
    eff_recotIso.SetFillColor(0)
    eff_recolIso.SetFillColor(0)
    eff_recoIDtIso.SetFillColor(0)
    eff_recoIDlIso.SetFillColor(0)
    eff_reco.SetMarkerStyle(2)
    eff_recoID.SetMarkerStyle(3)
    eff_recotIso.SetMarkerStyle(5)
    eff_recolIso.SetMarkerStyle(27)
    eff_recoIDtIso.SetMarkerStyle(4)
    eff_recoIDlIso.SetMarkerStyle(28)
    
    eff_reco.Write("eff_reco_truth")
    eff_ID.Write("eff_ID_reco")
    eff_tIso.Write("eff_tIso_reco")
    eff_lIso.Write("eff_lIso_reco")
    eff_IDtIso.Write("eff_IDtIso_reco")
    eff_IDlIso.Write("eff_IDlIso_reco")
    eff_recoID.Write("eff_recoID_truth")
    eff_recotIso.Write("eff_recotIso_truth")
    eff_recolIso.Write("eff_recolIso_truth")
    eff_recoIDtIso.Write("eff_recoIDtIso_truth")
    eff_recoIDlIso.Write("eff_recoIDlIso_truth")

    eff_reco.SetTitle(";True p_{T}^{#gamma} [GeV];Efficiency")
    
    c=r.TCanvas("effs","effs",800,600)
    c.SetGridy(1)
    c.SetLeftMargin(0.15)
    c.SetBottomMargin(0.15)
    eff_reco.Draw()
    eff_recoIDtIso.Draw("same")
    eff_recoIDlIso.Draw("same")
    eff_recoID.Draw("same")
    eff_recotIso.Draw("same")
    eff_recolIso.Draw("same")
    c.Update()
    
    axishist=eff_reco.GetPaintedGraph().GetHistogram()
    axishist.GetYaxis().SetTitle("Efficiency")
    axishist.GetYaxis().SetRangeUser(0,1)
    axishist.GetYaxis().SetLabelSize(0.05)
    axishist.GetXaxis().SetLabelSize(0.05)
    axishist.GetXaxis().SetTitleSize(0.06)
    axishist.GetYaxis().SetTitleSize(0.06)
    axishist.GetXaxis().SetTitleOffset(1.1)
    
    c.Update()
    r.gPad.Update()

    leg=r.TLegend(0.3,0.2,0.85,0.5)
    leg.AddEntry(eff_reco, "looseID")
    leg.AddEntry(eff_recoID, "looseID+tightID")
    leg.AddEntry(eff_recotIso, "looseID+tightIso")
    leg.AddEntry(eff_recolIso, "looseID+looseIso")
    leg.AddEntry(eff_recoIDtIso, "looseID+tightID+tightIso")
    leg.AddEntry(eff_recoIDlIso, "looseID+tightID+looseIso")
    leg.Draw()

    tl=r.TLatex()
    tl.SetNDC()
    tl.SetTextSize(0.03)
    tl.SetTextFont(82)
    tl.DrawLatex(0.6, 0.92,f"{f[13:-5]}")
    
    c.Print(f"{f[:-5]}_effs.pdf")


    # pt plot
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

