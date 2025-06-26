#!/usr/bin/bash

version="v3"

for inputfiles in ""; do #"_sig" "_mc" "_data"; do

    jsoninput=dataset_runnable/af_${version}${inputfiles}.json
    dataset_names=$(jq -r 'keys_unsorted[]' ${jsoninput} | sed s/_mc20[ade]/_mc20/g | sed s/_mc23[ad]/_mc23/g | uniq)

    for ds in ${dataset_names}; do

	if [[ $ds == *mc20 ]]; then
	    for campaign in "a" "d" "e" ""; do
		if [[ $ds == data* && campaign != "" ]]; then
		    continue
		fi
		
		if ! test -f /data/mhance/SUSY/ntuples/v3.1_condor/output_${ds}${campaign}.root; then
		    echo "${ds}${campaign} is missing"
		fi
	    done
	else
	    for campaign in "a" "d" ""; do
		if [[ $ds == data* && campaign != "" ]]; then
		    continue
		fi
		
		if ! test -f /data/mhance/SUSY/ntuples/v3.1_condor/output_${ds}${campaign}.root; then
		    echo "${ds}${campaign} is missing"
		fi
	    done
	fi	    
    done
done
