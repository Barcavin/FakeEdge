#!/bin/bash
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   KEY="${KEY:2}"
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

method=$method
data=$data
fuse=$fuse
[[ -z "${val}" ]] && val='0.05' || val="${val}"
[[ -z "${test}" ]] && test='0.1' || test="${test}"
[[ -z "${csv_dir}" ]] && csv_dir='results' || csv_dir="${csv_dir}"

if [ "$method" = "SEAL" ]
then
    echo "Run SEAL"
    bash seal.sh $method $data $fuse $val $test $csv_dir #&& python -c "import wandb;wandb.init(project='FakeEdge', entity='kevindong')"
elif [ "$method" = "GCN" ] || [ "$method" = "SAGE" ] || [ "$method" = "GIN" ] || [ "$method" = "gMPNN" ]
then
    echo "RUN $method"
    bash seal.sh $method $data $fuse $val $test $csv_dir
elif [ "$method" = "WalkPool" ]
then
    echo "RUN $method"
    bash walkpooling.sh $data $fuse $val $test $csv_dir
elif [ "$method" = "PLNLP" ]
then
    echo "RUN $method"
    bash plnlp.sh $data $fuse $val $test $csv_dir
fi