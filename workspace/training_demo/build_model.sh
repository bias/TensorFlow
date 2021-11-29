#!/bin/bash

MODEL=$1
CHECK=$2

BATCH=$(grep batch_size ${MODEL}/pipeline.config | head -n 1 | sed 's/.*batch_size:\s*\(\d*\)/\1/')
REP=$(grep replicas_to_aggregate ${MODEL}/pipeline.config | sed 's/.*replicas_to_aggregate:\s*\(\d*\)/\1/')
WARM=$(grep warmup_steps ${MODEL}/pipeline.config | sed 's/.*warmup_steps:\s*\(\d*\)/\1/')
TOTAL=$(grep total_steps ${MODEL}/pipeline.config | sed 's/.*total_steps:\s*\(\d*\)/\1/')

echo batch \"$BATCH\"
echo replicas \"$REP\"
echo warmup \"$WARM\"
echo total \"$TOTAL\"

python3 model_main_tf2.py \
  --checkpoint_every_n $CHECK \
  --model_dir=$MODEL \
  --pipeline_config_path=$MODEL/pipeline.config \
  > $MODEL/batch_${BATCH}_replicas_${REP}_steps_${WARM}:${TOTAL}.log 2>&1 &!
