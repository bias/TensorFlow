#!/usr/bin/env python3

import os
import sys
import argparse

import tensorflow as tf


my_parser = argparse.ArgumentParser(description='process oly lifting mp4 videos')
my_parser.add_argument('--model', required=True, help='model directory')
args = my_parser.parse_args()

saved_model_tflite = "%s/saved_model_tflite" % args.model

if not os.path.exists(saved_model_tflite):
    print("%s doesn't exist" % saved_model_tflite)
    print("failure in \'export_lite_model.sh\' ???")
    sys.exit(1)

base_file_name = os.path.basename(args.model)
out_file = "exported-models/%s/%s.tflite" % (base_file_name, base_file_name)


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_tflite) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open(out_file, 'wb') as f:
  f.write(tflite_model)
