#!/bin/bash

MODEL=$1

WARM=$(grep warmup_steps ${MODEL}/pipeline.config | sed 's/.*warmup_steps:\s*\(\d*\)/\1/')
TOTAL=$(grep total_steps ${MODEL}/pipeline.config | sed 's/.*total_steps:\s*\(\d*\)/\1/')

OUT=out/$(basename $MODEL)

mkdir -p $OUT

./run_checkpoint.py \
  --model $MODEL \
  --annotations annotations/multi_angle_3 \
  --image_dir images/multi_angle_3/test \
  --out_dir $OUT
