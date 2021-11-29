#!/bin/bash

MODEL=$1

WARM=$(grep warmup_steps ${MODEL}/pipeline.config | sed 's/.*warmup_steps:\s*\(\d*\)/\1/')
TOTAL=$(grep total_steps ${MODEL}/pipeline.config | sed 's/.*total_steps:\s*\(\d*\)/\1/')

echo warmup \"$WARM\"
echo total \"$TOTAL\"

python3 exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path $MODEL/pipeline.config \
  --trained_checkpoint_dir $MODEL \
  --output_directory exported-models/$(basename $MODEL)_${WARM}_${TOTAL}
