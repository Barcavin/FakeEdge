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

if [ "$method" = "SEAL" ]
then
    echo "Run SEAL"
    bash seal.sh $method $data $fuse #&& python -c "import wandb;wandb.init(project='FakeEdge', entity='kevindong')"
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