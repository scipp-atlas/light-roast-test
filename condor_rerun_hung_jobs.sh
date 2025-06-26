for i in $(condor_q | grep "mhance ID" | awk '{print $3}'); do
    tag=$(grep $i condor/*.log | egrep -v "Wtau|Wmu|Znu" | head -1 | awk '{print $1}' | sed s="condor/"==g | sed s=".log:000"==g | sed s="picontuple."==g)
    if [[ ${tag} == "" ]]; then
	continue
    fi
    #echo $tag
    #ls condor/job_${tag}.sub
    condor_rm $i
    condor_submit condor/job_${tag}.sub
    sleep 1
done
