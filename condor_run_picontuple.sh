#!/bin/bash

echo "Starting $1 $2 $3 $4 $5 on host $HOSTNAME"

ds=$1
jsoninput=$2
outputarea=$3
N=$4

SCRATCH=${PWD}

echo "Copying files"

cp -p /home/mhance/light-roast/picontuple.py .
cp -p ${jsoninput} .
jsoninput="${jsoninput##*/}"

echo "Setting environment"

# this is super slow
#source /home/mhance/light-roast-env/bin/activate

# this doesn't work, still points to home area
#cp /home/mhance/light-roast-env.tar.gz .
#tar -zxf light-roast-env.tar.gz
#source light-roast-env/bin/activate

# this works much better
source /data/mhance/light-roast-env/bin/activate

which python3

echo "Executing"

python3 -v picontuple.py --dataset ${ds} --infile ${jsoninput} -o . -n ${N}

echo "Done"

ls -ltrh

echo "Transferring"

# transfer output
mv output_${ds}.root ${outputarea}

echo "Cleanup"

# cleanup
rm ${jsoninput}
rm picontuple.py

echo "Exit"
