#!/bin/bash
data=$1
fuse=$2

echo "Run $data : $fuse"

if [[ $data = cora ]] || [[ $data = citeseer ]] || [[ $data = pubmed ]]
then
    python PLNLP/main.py --batch_size=128 --data_name=$data --drnl=true --dynamic=true --encoder=GATv2 --fusion=$fuse --runs=10 --use_node_feats=true --csv=true
else
    python PLNLP/main.py --batch_size=128 --data_name=$data --drnl=true --dynamic=true --encoder=GATv2 --fusion=$fuse --runs=10 --train_node_emb=false --use_node_feats=false --csv=true
fi