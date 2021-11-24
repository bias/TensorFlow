#!/bin/bash

MODEL=$1
CHECK=$2
BATCH=$3
REP=$4
WARM=$5
TOTAL=$6

python3 model_main_tf2.py \
  --checkpoint_every_n $CHECK \
  --model_dir=$MODEL \
  --pipeline_config_path=$MODEL/pipeline.config \
  > $MODEL/batch_${BATCH}_replicas_${REP}_steps_${WARM}:${TOTAL}.log 2>&1 &!
