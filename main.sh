#!/bin/bash
method=$1
data=$2
fuse=$3

if [ "$method" = "SEAL" ]
then
    echo "Run SEAL"
elif [ "$method" = "GCN" ] || [ "$method" = "SAGE" ] || [ "$method" = "GIN" ]
then
    echo "RUN $method"
elif [ "$method" = "WalkPool" ]
then
    echo "RUN $method"
elif [ "$method" = "PLNLP" ]
then
    echo "RUN $method"
    bash plnlp.sh $data $fuse
fi