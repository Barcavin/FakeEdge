#!/bin/bash
method=$1
name=$2
fuse=$3
val=$4
test=$5
csv_dir=results

if [[ $method = SEAL ]]
then
    model=DGCNN
else
    model=$method
fi



if [[ $model = gMPNN ]]
then
    echo "Run gMPNN"
    if [[ $name = cora ]]
    then
        echo "Run cora"
        python SEAL/seal_link_pred.py --dataset $name --num_hops 1 --use_feature --hidden_channels 256 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test --batch_size=18
    elif [[ $name = citeseer ]]
    then
        echo "Run citeseer"
        python SEAL/seal_link_pred.py --dataset $name --num_hops 1 --use_feature --hidden_channels 256 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test --batch_size=18
    elif [[ $name = pubmed ]]
        then
        echo "Run pubmed"
        python SEAL/seal_link_pred.py --dataset $name --num_hops 1 --use_feature --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test --batch_size=40
    elif [[ $name = Ecoli ]]
        then
        echo "Run $name"
        python SEAL/seal_link_pred.py --dataset $name --num_hops 1 --hidden_channels 32 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test --batch_size=2
    elif [[ $name = PB ]]
        then
        echo "Run $name"
        python SEAL/seal_link_pred.py --dataset $name --num_hops 1 --hidden_channels 32 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test --batch_size=12
    else
        echo "Run Wo"
        python SEAL/seal_link_pred.py --dataset $name --num_hops 1 --hidden_channels 128 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test --batch_size=18
    fi
elif [[ $name = cora ]]
then
    echo "Run cora"
    python SEAL/seal_link_pred.py --dataset $name --num_hops 3 --use_feature --hidden_channels 256 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test
elif [[ $name = citeseer ]]
then
    echo "Run citeseer"
    # not using --use_feature here in their github repo
    python SEAL/seal_link_pred.py --dataset $name --num_hops 3 --hidden_channels 256 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test
elif [[ $name = pubmed ]]
    then
    echo "Run pubmed"
    python SEAL/seal_link_pred.py --dataset $name --num_hops 3 --use_feature --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test
else
    echo "Run Wo"
    python SEAL/seal_link_pred.py --dataset $name --num_hops 2 --hidden_channels 128 --runs 10 --fuse=$fuse --model=$model --dynamic_train --dynamic_val --dynamic_test --csv=$csv_dir --val-ratio=$val --test-ratio=$test
fi