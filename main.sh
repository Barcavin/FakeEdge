#!/bin/bash
method=$1
data=$2
fuse=$3

if [ "$method" = "SEAL" ]
then
    echo "Run SEAL"
    bash seal.sh $method $data $fuse
elif [ "$method" = "GCN" ] || [ "$method" = "SAGE" ] || [ "$method" = "GIN" ]
then
    echo "RUN $method"
    bash seal.sh $method $data $fuse
elif [ "$method" = "WalkPool" ]
then
    echo "RUN $method"
    bash walkpooling.sh $data $fuse
elif [ "$method" = "PLNLP" ]
then
    echo "RUN $method"
    bash plnlp.sh $data $fuse
fi