#!/bin/bash
source $1

declare -a MODELS=("dbmdz/distilbert-base-turkish-cased" "dbmdz/bert-base-turkish-cased" )

for MODEL in ${MODELS[@]}
do
    python train-nli.py --model $MODEL \
                        --batch_size 32 \
                        --output_dir /home/emrecan/tez/zeroshot-turkish/models \
done