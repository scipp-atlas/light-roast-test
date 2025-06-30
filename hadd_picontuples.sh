#!/usr/bin/bash

source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
source $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh "root 6.34.04-x86_64-el9-gcc13-opt"

version="v3"

N=5

for inputfiles in ""; do #"_sig" "_mc" "_data"; do

    jsoninput=../light-roast/dataset_runnable/af_${version}${inputfiles}.json
    dataset_names=$(jq -r 'keys_unsorted[]' ${jsoninput} | sed s/_mc20[ade]/_mc20/g | sed s/_mc23[ad]/_mc23/g | uniq)

    for ds in ${dataset_names}; do

	if [[ $ds == data_20* ]]; then
	   continue
	fi
    
        echo "Working on ${ds}"

	hadd -f /data/mhance/SUSY/ntuples/v3.1_condor_2/output_${ds}.root /data/mhance/SUSY/ntuples/v3.1_condor_2/output_${ds}*.root &

	if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
	    # now there are $N jobs already running, so wait here for any job
	    # to be finished so there is a place to start next one.
	    wait -n
	fi

    done
done
