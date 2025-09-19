#!/usr/bin/bash

source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
source $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.34.04-x86_64-el9-gcc13-opt"

af_version="v3"
nt_version="v3.Y"

N=5

rm -rf /data/mhance/SUSY/ntuples/${nt_version}/processor

for inputfiles in ""; do #"_sig" "_mc" "_data"; do

    #jsoninput=../light-roast/dataset_runnable/af_${af_version}${inputfiles}.json
    jsoninput=/data/kratsg/radiative-decays/af_${af_version}.json
    dataset_names=$(jq -r 'keys_unsorted[]' ${jsoninput} | sed s/_mc20[ade]/_mc20/g | sed s/_mc23[ad]/_mc23/g | uniq)

    for ds in ${dataset_names}; do

	if [[ $ds == data_20* ]]; then
	   continue
	fi
    
        echo "Working on ${ds}"

	#hadd -f /data/mhance/SUSY/ntuples/v3.1_condor_2/output_${ds}.root /data/mhance/SUSY/ntuples/v3.1_condor_2/output_${ds}*.root &
	rm -f /data/mhance/SUSY/ntuples/${nt_version}/output_${ds}.root 
	mkdir -p /data/mhance/SUSY/ntuples/${nt_version}/campaigns
	mv /data/mhance/SUSY/ntuples/${nt_version}/output_${ds}*.root /data/mhance/SUSY/ntuples/${nt_version}/campaigns
	hadd -f /data/mhance/SUSY/ntuples/${nt_version}/output_${ds}.root /data/mhance/SUSY/ntuples/${nt_version}/campaigns/output_${ds}*.root &

	if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
	    # now there are $N jobs already running, so wait here for any job
	    # to be finished so there is a place to start next one.
	    wait -n
	fi

    done
done
