#!/usr/bin/env python3

import os
import pathlib
import argparse
import sys

import cv2
import tensorflow as tf

import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


detection_thresh = 0.3
max_boxes = 8

my_parser = argparse.ArgumentParser(description='process oly lifting mp4 videos')
my_parser.add_argument('--model', required=True, help='model directory')
my_parser.add_argument('--annotations', required=True, help='annotations directory')
my_parser.add_argument('--image_dir', required=True, help='generate processed image frames')
my_parser.add_argument('--out_dir', required=True, help='processing output dir')
args = my_parser.parse_args()


if not os.path.exists(args.image_dir):
    print("%s doesn't exist" % args.image_dir)
    sys.exit(1)
    
images = os.listdir(args.image_dir)

if not os.path.exists(args.out_dir):
    print("%s doesn't exist" % args.out_dir)
    sys.exit(1)


if not os.path.exists(args.model):
    print("%s doesn't exist" % args.model)
    sys.exit(1)

if not os.path.exists(args.annotations):
    print("%s doesn't exist" % args.annotations)
    sys.exit(1)

# XXX this assumes model file and annotation layout
PATH_TO_MODEL_DIR = args.model
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
PATH_TO_ANNOTATION = args.annotations
PATH_TO_LABELS = PATH_TO_ANNOTATION + "/label_map.pbtxt"


print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def main():

    for image in images:

        suffexless_file_name = image[:-4]
        suffex = image[-4:]
        if suffex != ".jpg":
            continue

        image_path = args.image_dir + "/" + image
        image_out_path = "%s/%s_out.jpg" % (args.out_dir, suffexless_file_name)

        print("Converting %s to %s" % (image_path, image_out_path))

        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=max_boxes,
                min_score_thresh=detection_thresh,
                agnostic_mode=False)

        cv2.imwrite(image_out_path, image_np_with_detections)



#
# main
#
if __name__ == "__main__":
    main()

