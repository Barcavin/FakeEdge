#!/bin/bash
method=$1
name=$2
fuse=$3

if [[ $method = SEAL ]]
then
    model=DGCNN
else
    model=$method
fi

if [[ $name = cora ]]
then
    echo "Run cora"
    python seal_link_pred.py --dataset $name --num_hops 3 --use_feature --hidden_channels 256 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv
elif [[ $name = citeseer ]]
then
    echo "Run citeseer"
    # not using --use_feature here in their github repo
    python seal_link_pred.py --dataset $name --num_hops 3 --hidden_channels 256 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv
elif [[ $name = pubmed ]]
    then
    echo "Run pubmed"
    python seal_link_pred.py --dataset $name --num_hops 3 --use_feature --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv
else
    echo "Run Wo"
    python seal_link_pred.py --dataset $name --num_hops 2 --hidden_channels 128 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv
fi