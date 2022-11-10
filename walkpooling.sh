#!/bin/bash
name=$1
fuse=$2
val=$3
test=$4
csv_dir=$5

time=$(date +%s)

if [[ $data = cora ]] || [[ $data = citeseer ]] || [[ $data = pubmed ]]
then
  for i in 1 2 3 4 5 6 7 8 9 10
  do
    python WalkPooling/main.py --time $time --seed $i --data-name $name --drnl 0 --init-attribute none --init-representation none --embedding-dim 32 --practical-neg-sample true --fuse=$fuse --csv=$csv_dir --val-ratio=$val --test-ratio=$test
  done
else
  for i in 1 2 3 4 5 6 7 8 9 10
  do
      python WalkPooling/main.py --time $time --seed $i --use-splitted false --data-name $name --drnl 0 --init-attribute ones --init-representation none --embedding-dim 16 --practical-neg-sample true --fuse=$fuse --csv=$csv_dir --val-ratio=$val --test-ratio=$test
  done
fi
