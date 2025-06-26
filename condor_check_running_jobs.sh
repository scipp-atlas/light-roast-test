jobs=$(condor_q | grep "mhance " | awk '{print $2}')
njobs=$(echo $jobs | wc | awk '{print $2}')

echo "Total jobs: ${njobs}"
echo "TaskID    Job name"
echo "----------------------------------------"
for i in ${jobs}; do
    echo "$i";
done
