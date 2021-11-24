#!/usr/bin/env python3

import os
import sys
import argparse
import warnings
import time
import csv

import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

detection_thresh = 0.1
max_boxes = 8
label_offset = 1

frames_to_process = 0
frame_count = 0
frame_rate = 0
frame_x = 0
frame_y = 0


# Create the parser
my_parser = argparse.ArgumentParser(description='process oly lifting mp4 videos')
my_parser.add_argument('--mp4', required=True, help='mp4 video')
my_parser.add_argument('--model', required=True, help='model directory')
my_parser.add_argument('--annotations', required=True, help='annotations directory')
my_parser.add_argument('--out_dir', required=False, help='processing output dir')
#my_parser.add_argument('--video', nargs='?', const=True, default='file_name', required=False, help='processing video')
#my_parser.add_argument('--images', nargs='?', const=True, default='dir_name', required=False, help='generate processed image frames')
my_parser.add_argument('--video', nargs='?', const=True, default=None, required=False, help='processing video')
my_parser.add_argument('--images', nargs='?', const=True, default=None, required=False, help='generate processed image frames')
my_parser.add_argument('--ellipse', nargs='?', const=True, default=None, required=False, help='generate contrast ellipse')
my_parser.add_argument('--frames', required=False, help='number of frames to process')
args = my_parser.parse_args()


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

global category_index
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

end_time = time.time()
elapsed_time = end_time - start_time

print('Done! Took {} seconds'.format(elapsed_time))


suffexless_file_name = ""

def main():


    global suffexless_file_name
    if not os.path.exists(args.mp4):
        print("%s doesn't exist" % args.mp4)
        sys.exit(1)

    if args.out_dir:
        if not os.path.exists(args.out_dir):
            sys.exit(1)
            print("%s doesn't exist" % args.out_dir)
        base_mp4_file_name = os.path.basename(args.mp4)
        suffexless_file_name = base_mp4_file_name[:-4]
        csv_file_name = args.out_dir + "/" + suffexless_file_name + "_trajectory.csv"
        mp4_out_file_name = args.out_dir + "/" + suffexless_file_name + "_out.mp4"
    else:
        relative_mp4_file_name = args.mp4
        suffexless_file_name = relative_mp4_file_name[:-4]
        csv_file_name = suffexless_file_name + ".csv"
        mp4_out_file_name = suffexless_file_name + "_out" + ".mp4"

    global frames_to_process
    if args.frames:
        frames_to_process = int(args.frames)

    if args.ellipse:
        print("ellipse %s" % args.ellipse)
    else:
        print("f ellipse %s" % args.ellipse)

    if args.video:
        print("video %s" % args.video)
    else:
        print("f video %s" % args.video)
        sys.exit(1)

    with open(csv_file_name, 'w', newline='') as csvfile:
        trajectory_writer = csv.writer(csvfile, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
        detect_video(args.mp4, mp4_out_file_name, trajectory_writer)




# Predict video from folder
def detect_video(path, output_path, trajectory_writer):

    # Read the video
    print("reading video %s" % path)
    vidcap = cv2.VideoCapture(path)
    frame_read, image = vidcap.read()

    global frame_count
    global frame_rate
    global frame_x
    global frame_y
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))
    frame_x = int(vidcap.get(3))
    frame_y = int(vidcap.get(4))
    print("frame :: count %s, rate %s, dim_x %s, dim_y %s" % (frame_count, frame_rate, frame_x, frame_y))

    # Set output video writer with codec
    if args.video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_x, frame_y))
    
    print("processing", end='', flush=True)
    # Iterate over frames and pass each for prediction
    current_frame = 0
    while frame_read:

        print('.', end='', flush=True)

        output_file = detect_image(image, current_frame, trajectory_writer)

        if args.video:
            out.write(output_file)

        if args.images:
            # FIXME FIXME update path and file names
            cv2.imwrite("%s/%s_frame_%s.jpg" % (args.out_dir, suffexless_file_name, current_frame), output_file)

        # Read next frame
        frame_read, image = vidcap.read()
        current_frame += 1

        # XXX early exit
        if frames_to_process != 0:
            if current_frame >= frames_to_process:
                frame_read = False

    # Release video file when we're ready
    if args.video:
        out.release()


def detect_image(image, current_frame, trajectory_writer):

    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    plates = process_frame(detections, current_frame, trajectory_writer)

    # FIXME WIP
    if args.ellipse and current_frame == 0:
        cp = plates['first_close_plates']
        print(frame_y, frame_x)
        print(image.shape)
        print(cp)
        fp = plates['first_far_plates']
        print(fp)

        buf = 10

        # XXX invert y dim
        cp_y_min = max( int(frame_y-cp['y_max']) - buf, 0)
        cp_y_max = min( int(frame_y-cp['y_min']) + buf, frame_y)
        cp_x_min = max( int(cp['x_min']) - buf, 0)
        cp_x_max = min( int(cp['x_max']) + buf, frame_x)
        print("%s %s %s %s" % (cp_y_min, cp_y_max, cp_x_min, cp_x_max))
        cropped_close_plates = image[cp_y_min:cp_y_max, cp_x_min:cp_x_max].copy()
        cv2.imwrite("turd_close.jpg", cropped_close_plates) 
        # Convert to graycsale
        gray_ccp = cv2.cvtColor(cropped_close_plates, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        blur_ccp = cv2.GaussianBlur(gray_ccp, (3,3), 0)
        # Sobel Edge Detection
        sobelxy = cv2.Sobel(src=blur_ccp, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        canny = cv2.Canny(image=blur_ccp, threshold1=100, threshold2=200) # Canny Edge Detection
        cv2.imwrite("turd_close_canny.jpg", canny) 
        # Find contours (modifies source image!)
        contours,hierarchy = cv2.findContours(canny, 1, 2)
        thresh_contour = cropped_close_plates.copy()
        cv2.drawContours(image=thresh_contour, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite("turd_close_canny_contour.jpg", thresh_contour) 
        flat_list = np.concatenate(contours)
        print(len(flat_list))
        ellipse = cv2.fitEllipse(flat_list)
        canny_ellipse = cv2.ellipse(cropped_close_plates, ellipse, (0,255,0), 2)
        cv2.imwrite("turd_close_canny_ellipse.jpg", canny_ellipse) 
        
        
        #cv2.calibrateCamera()

#        ret,thresh = cv2.threshold(sobelxy,127,255,cv2.THRESH_BINARY)
#        thresh_image = thresh.astype(np.uint8)
#        contours,hierarchy = cv2.findContours(thresh_image, 1, 2)
#        cnt = contours[0]
#        ellipse = cv2.fitEllipse(cnt)
#        sobelxy_ellipse = cv2.ellipse(sobelxy, ellipse, (0,255,0), 2)
#        cv2.imwrite("turd_close_sobelxy.jpg", sobelxy)
#        cv2.imwrite("turd_close_sobelxy_ellipse.jpg", sobelxy_ellipse)

        # XXX invert y dim
        fp_y_min = max( int(frame_y-fp['y_max']) - buf, 0)
        fp_y_max = min( int(frame_y-fp['y_min']) + buf, frame_y)
        fp_x_min = max( int(fp['x_min']) - buf, 0)
        fp_x_max = min( int(fp['x_max']) + buf, frame_x)
        print("%s %s %s %s" % (fp_y_min, fp_y_max, fp_x_min, fp_x_max))
        cropped_far_plates = image[fp_y_min:fp_y_max, fp_x_min:fp_x_max].copy()
        # Convert to graycsale
        gray_cfp = cv2.cvtColor(cropped_far_plates, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        blur_cfp = cv2.GaussianBlur(gray_cfp, (3,3), 0)
        # Sobel Edge Detection
        sobelxy = cv2.Sobel(src=blur_cfp, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        canny = cv2.Canny(image=blur_cfp, threshold1=100, threshold2=200) # Canny Edge Detection
        cv2.imwrite("turd_far_canny.jpg", canny) 
        # Find contours (modifies source image!)
        contours,hierarchy = cv2.findContours(canny, 1, 2)
        thresh_contour = cropped_far_plates.copy()
        cv2.drawContours(image=thresh_contour, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite("turd_far_canny_contour.jpg", thresh_contour) 
        flat_list = np.concatenate(contours)
        print(len(flat_list))
        ellipse = cv2.fitEllipse(flat_list)
        canny_ellipse = cv2.ellipse(cropped_far_plates, ellipse, (0,255,0), 2)
        cv2.imwrite("turd_far_canny_ellipse.jpg", canny_ellipse) 



    # Perform visualization on output image/frame 
    if args.video or args.images:
        viz_utils.visualize_boxes_and_labels_on_image_array(
          image,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy()+label_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=max_boxes,
          min_score_thresh=detection_thresh, 
          agnostic_mode=False)
    
    # Return the image
    return image


def process_frame(detections, current_frame, trajectory_writer):
    box_map = {
        '1': 'close_plates',
        '2': 'far_plates',
        '3': 'bar_end',
        '4': 'bar_end_sleeve'
    }
    box_map_count = {
        'close_plates': 0,
        'far_plates': 0,
        'bar_end': 0,
        'bar_end_sleeve': 0
    }

    first_close_plates = None
    first_far_plates = None

    count = 0
    #print(detections['detection_boxes'][0])
    #print(detections['detection_scores'][0])
    #print(detections['detection_classes'][0])
    for box in detections['detection_boxes'][0].numpy():
        box_score = detections['detection_scores'][0].numpy()[count]
        if box_score >= detection_thresh:
            box_class = int(detections['detection_classes'][0].numpy()[count]) + label_offset
            #box_label = box_map[str(box_class)]
            #frame_rate

            #print("frame_x %s, frame_y %s, box %s %s %s %s" % (frame_x, frame_y, box[0], box[1], box[2], box[3])) 
            # renormalize into pixel size
            y_max = frame_y - box[0] * frame_y
            y_min = frame_y - box[2] * frame_y
            x_min = box[1] * frame_x
            x_max = box[3] * frame_x
            #print("\nbox %s %s %s %s" % (y_max, y_min, x_min, x_max))

            # XXX highest score box is returned first, get best close/far plates estimate
            if box_map[str(box_class)] == 'close_plates' and first_close_plates == None:
                first_close_plates =  {
                    'y_max': y_max,
                    'x_max': x_max,
                    'y_min': y_min,
                    'x_min': x_min,
                }

            if box_map[str(box_class)] == 'far_plates' and first_far_plates == None:
                first_far_plates =  {
                    'y_max': y_max,
                    'x_max': x_max,
                    'y_min': y_min,
                    'x_min': x_min,
                }

            xmean = (x_max + x_min) / 2;
            ymean = (y_max + y_min) / 2;
            xlen = x_max - x_min;
            ylen = y_max - y_min;
            ratio = xlen / ylen;

            trajectory_writer.writerow([current_frame, box_class, box_score, y_min, x_min, y_max, x_max, xmean, ymean, xlen, ylen, ratio])

        count += 1

    return {'first_close_plates': first_close_plates, 'first_far_plates': first_far_plates}

#
# main
#
if __name__ == "__main__":
    main()

