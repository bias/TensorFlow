#!/bin/bash

MODEL=$1

WARM=$(grep warmup_steps ${MODEL}/pipeline.config | sed 's/.*warmup_steps:\s*\(\d*\)/\1/')
TOTAL=$(grep total_steps ${MODEL}/pipeline.config | sed 's/.*total_steps:\s*\(\d*\)/\1/')

echo warmup \"$WARM\"
echo total \"$TOTAL\"

TEMP=$(basename $MODEL)_${WARM}_${TOTAL}_ftlite
OUT=$(basename $MODEL)_${WARM}_${TOTAL}

echo
echo "*** Generating tflite graph ***"
echo

python3 export_tflite_graph_tf2.py \
  --pipeline_config_path $MODEL/pipeline.config \
  --trained_checkpoint_dir $MODEL \
  --output_directory exported-models/$TEMP

mv exported-models/$TEMP/saved_model exported-models/$OUT/saved_model_tflite
rmdir exported-models/$TEMP

echo
echo "*** Exporting tflite model ***"
echo

./convert_to_tflite.py --model exported-models/$OUT
