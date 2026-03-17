import json

print(f"{'Year':4s}    {'RunNo':8s}     {'EvtNo':12s}")
for ABCD in ["TL","LT","LL"]:
    years=[]
    runnos=[]
    evtnos=[]
    for year in [15,16,17,18]: #,22,23,24]:
        with open(f"ABCD_results_4/output_data_20{year}_ABCD_tightID_hybridCOIso_LoosePrime4.json",'r') as j:
            data = json.load(j)
            for i in range(len(data["SR"]["0L-mT-low"][ABCD]["runNumbers"])):
                years.append(2000+year)
                runnos.append(data["SR"]["0L-mT-low"][ABCD]["runNumbers"][i])
                evtnos.append(data["SR"]["0L-mT-low"][ABCD]["eventNumbers"][i])

    print(ABCD)
    for y,r,e in zip(years,runnos,evtnos):
        print(f"{y:4d} {r:8d} {e:12d}")
    

