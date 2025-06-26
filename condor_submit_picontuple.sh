#!/usr/bin/bash

version="v3"
outputarea=/data/mhance/SUSY/ntuples/v3.1_condor
jsoninput=${PWD}/dataset_runnable/af_${version}.json
dataset_names=$(jq -r 'keys_unsorted[]' ${jsoninput})

for ds in ${dataset_names}; do
    if [[ $1 != "" ]]; then
        if [[ $ds != *${1}* ]]; then
            continue
        fi
    fi

    N=4
    mem=8
    if [[ $ds == Wtaunugamma* ]]; then
	N=16
	mem=16
    elif [[ $ds == N2_* || $ds == yjets_* || $ds == Sh_2212_* || $ds == Pythia8EvtGen* || $ds == tqgamma* || $ds == Py8_gammajet* || $ds == PhPy8EG_* || $ds == PowhegPythia8EvtGen_* ]]; then
	N=1
	mem=4
    fi
    
    echo "Working on ${ds} with ${N} cores and ${mem} GB of memory"

    # add these in if needed for debugging, otherwise I suspect it just slows things down
    #stream_output = True
    #stream_error = True

    echo """
Universe = vanilla

Output = condor/picontuple.${ds}.\$(Cluster).\$(Process).out
Error = condor/picontuple.${ds}.\$(Cluster).\$(Process).err
Log = condor/picontuple.${ds}.log

Executable = condor_run_picontuple.sh
Arguments = ${ds} ${jsoninput} ${outputarea} ${N}

batch_name = ${ds}

request_memory = ${mem}GB
request_cpus = ${N}

+queue=\"short\"
+ALLOW_MWT2=True

Queue 1

""" > condor/job_${ds}.sub

    
    condor_submit condor/job_${ds}.sub
done
