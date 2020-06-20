# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import win32gui
import win32con
import keyboard
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D, E, Z, X, R, C, UP, ONE, SPACE
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
from playsound import playsound
import cv2
#autosteer START
import math
#select the region of interest for the detected edges
def roi(image, polygons):
      mask = np.zeros_like(image)
      cv2.fillPoly(mask, polygons, 255)
      masked = cv2.bitwise_and(image, mask)
      return masked

#display the lines on the screen
def display_line(image, line):
      line_image = np.zeros_like(image)
      if lines is not None:
            for line in lines:
                  x1, y1, x2, y2 = line[0]
                  cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),10)
      return line_image

#processing image for detecting edge using canny edge detection and blur the image using gaussian blur
def proceesed_img(original_image):
            proceesed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            proceesed_img = cv2.GaussianBlur(proceesed_img,(5,5), 0)
            proceesed_img = cv2.Canny(proceesed_img, threshold1 =150, threshold2 = 163 )
            #TRUCKONLY these polygon repressent the data point within with the pixel data are selected for lane detection
            #cockpit view
            polygons = np.array([[200,345],[650,345],[650,380],[200,380]])
            proceesed_img = roi(proceesed_img, [polygons])
            return proceesed_img

#this funtions sends the input to the game which is running on left side of screen
def straight():
      ReleaseKey(A)
      ReleaseKey(D)
      
def little_left():
      #indicate start
      #PressKey(Z)
      #ReleaseKey(Z)
      #indicate end
      PressKey(A)
      time.sleep(0.03)
      ReleaseKey(A)
      time.sleep(0.01)
      
def little_right():
      #indicate start
      #PressKey(X)
      #ReleaseKey(X)
      #indicate end
      PressKey(D)
      time.sleep(0.03)
      ReleaseKey(D)
      time.sleep(0.01)
#autosteer END


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# # Model preparation 
# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('./tensorflow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# define variable for later
x = False
y = False
apx_stored = 0.0
autosteerEnabled = True
directionLeft = 0
directionRight = 0
directionStraight = 1
cars = 1

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)



with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
      #200
      screen = cv2.resize(grab_screen(region=(536,225,1056,530)), (1056,530))
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      for i,b in enumerate(boxes[0]):

        # Main detection 1 person (mop up) 3 car   6 bus  8 truck    7 train (tanker)  37 sports ball (tanker)   11 fire hydrant (weird trucks)   73 laptop (tankers)  61 cake (buses sometimes)  77 cell phone (some cars from TRUCKONLY perspective)
        if classes[0][i] == 1 or classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8 or classes[0][i] == 7 or classes[0][i] == 37 or classes[0][i] == 11 or classes[0][i] == 73 or classes[0][i] == 61 or classes[0][i] == 77:
          cars = 1
          if scores[0][i] >= 0.35:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*1056),int(mid_y*530)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # TRUCKONLY midx right has been reduced across all & apx dist raised across cc adjusts not friction code. 
            # If I see something between distances 0.5 and 0.7 within my AOI, I should slow
            if apx_distance <=0.8 and apx_distance >0.6:
              if mid_x > 0.4 and mid_x < 0.5:
                if x == True or y == True:
                  ReleaseKey(W)
                  PressKey(S)

            # If I see something closer than distance 0.5 within my AOI, friction brake then re-engage tacc
            elif apx_distance <=0.6 and apx_distance >=0.1:
              if mid_x > 0.4 and mid_x < 0.5:
                if x == True or y == True:
                  if apx_stored != apx_distance:
                    playsound('fcws.wav')
                    ReleaseKey(W)
                    PressKey(SPACE)
                    ReleaseKey(SPACE)
                    time.sleep(1.5)
                    PressKey(SPACE)
                    ReleaseKey(SPACE)
                    PressKey(ONE)
                    ReleaseKey(ONE)
                    PressKey(UP)
                    time.sleep(0.2)
                    ReleaseKey(UP)
                    PressKey(R)
                    ReleaseKey(R)
                    apx_stored = apx_distance
                    time.sleep(2)

            # If I see something greater than distance 0.7 in my AOI, accelerate
            elif apx_distance >0.8:
              if mid_x > 0.4 and mid_x < 0.5:
                if x == True or y == True:
                  ReleaseKey(S)
                  PressKey(W)
                  
        # Traffic light detection
        '''elif classes[0][i] == 10:
          cars = 0
          if scores[0][i] >= 0.5:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*1056),int(mid_y*530)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if apx_distance <=1.0:
              if x == True or y == True:
                if apx_stored != apx_distance and cars == 0:
                  # Wait before applying friction brakes as we see it early
                  ReleaseKey(W)
                  playsound('warning.wav')
                  time.sleep(2)
                  # Apply friction brakes when closer for 2 secs, check still engaged
                  if x == True or y == True and cars == 0:
                    PressKey(SPACE)
                    ReleaseKey(SPACE)
                    time.sleep(2)
                    # Release friction brakes, reset view and cancel AP
                    PressKey(SPACE)
                    ReleaseKey(SPACE)
                    PressKey(ONE)
                    ReleaseKey(ONE)
                    apx_stored = apx_distance
                    x = False
                    y = False
                    playsound('off.wav')

        # Stop sign detection
        elif classes[0][i] == 13:
          cars = 0
          if scores[0][i] >= 0.5:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*1056),int(mid_y*530)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if apx_distance <=1.0:
              if x == False or y == False and cars == 0:
                if apx_stored != apx_distance and cars == 0:
                  playsound('warning.wav')
                  time.sleep(2)
                  apx_stored = apx_distance'''
      
      cv2.imshow('window',cv2.resize(image_np,(1056,530)))

      if keyboard.is_pressed('z'):
        if x == True:  # lane change to the left 
                  autosteerEnabled = False
                  PressKey(A)
                  time.sleep(0.5)
                  ReleaseKey(A)
                  PressKey(D)
                  time.sleep(0.6)
                  ReleaseKey(D)
                  PressKey(Z)
                  ReleaseKey(Z)
                  autosteerEnabled = True

      elif keyboard.is_pressed('x'):
        if x == True:  # lane change to the right
                  autosteerEnabled = False
                  PressKey(D)
                  time.sleep(0.5)
                  ReleaseKey(D)
                  PressKey(A)
                  time.sleep(0.6)
                  ReleaseKey(A)
                  PressKey(X)
                  ReleaseKey(X)
                  autosteerEnabled = True

      elif keyboard.is_pressed('n'):
        if y == False:
          y = True
          x = False
          playsound('on-tacc.wav')
        elif y == True:
          y = False
          playsound('off.wav')

      '''elif keyboard.is_pressed('c'):
        if x == False:
          x = True
          y = False
          playsound('on.wav')
        elif x == True:
          x = False
          playsound('off.wav')'''

      #autosteer START
      if x == True:
        new_image = proceesed_img(screen)
        lines = cv2.HoughLinesP(new_image, 2 ,np.pi/180, 10,np.array([]), minLineLength = 0.005, maxLineGap = 50)
        left_coordinate = []
        right_coordinate = []
        
        if lines is not None:
              for line in lines:
                    x1,y1,x2,y2 = line[0]
                    slope = (x2-x1)/(y2-y1)
                    if slope<0:
                          left_coordinate.append([x1,y1,x2,y2])
                    elif slope>0:
                          right_coordinate.append([x1,y1,x2,y2])
              l_avg = np.average(left_coordinate, axis =0)
              r_avg = np.average(right_coordinate, axis =0)
              l =l_avg.tolist()
              r = r_avg.tolist()
              try:
                    #with the found slope and intercept, this is used to find the value of point x on both left and right line
                    #the center point is denoted by finding center distance between two lines
                    c1,d1,c2,d2 = r
                    a1,b1, a2,b2 = l
                    l_slope = (b2-b1)/(a2-a1)
                    r_slope = (d2-d1)/(c2-c1)
                    l_intercept = b1 - (l_slope*a1)
                    r_intercept = d1 - (r_slope*c1)
                    y=360
                    l_x = (y - l_intercept)/l_slope
                    r_x = (y - r_intercept)/r_slope
                    distance = math.sqrt((r_x - l_x)**2+(y-y)**2)
                    #line_center repressent the center point on the line
                    line_center = distance/2
                    center_pt =[(l_x+line_center)]
                    #TRUCKONLY  autosteer criteria for normal curves
                    f_l = [(l_x+(line_center*1.06))]
                    f_r = [(l_x+(line_center*0.02))]
                    #TRUCKONLY create a center point. Higher = left bias. Lower = right bias.
                    center_fixed =[373]
                    x_1 = int(l_x)
                    x_2 = int(r_x)
                    '''The logic behind this code is simple,
                    the center_fixed should be in the center_line.
                    means the cars is in center of the lane, if its get away from center,
                    then the left and right functions are used accordingly'''
                    #straight
                    if center_pt==center_fixed and autosteerEnabled == True:
                          straight()
                          directionLeft = 0
                          directionRight = 0
                          directionStraight = 1
                    #normal curves
                    elif center_fixed < f_r and autosteerEnabled == True:
                          little_right()
                          directionLeft = 0
                          directionRight = 1
                          directionStraight = 0
                    elif center_fixed > f_l and autosteerEnabled == True:
                          little_left()
                          directionLeft = 1
                          directionRight = 0
                          directionStraight = 0
                    #not sure
                    else:
                          straight()
                          directionLeft = 0
                          directionRight = 0
                          directionStraight = 1
              except:
                    #no lines
                    pass
                    if directionLeft == 1:
                      directionLeft = 1
                      directionRight = 0
                      directionStraight = 0
                      little_left()
                    elif directionRight == 1:
                      directionLeft = 0
                      directionRight = 1
                      directionStraight = 0
                      little_right()
                    elif directionStraight == 1:
                      directionLeft = 0
                      directionRight = 0
                      directionStraight = 1
                      straight()

        line_image = display_line(screen,lines)    
        combo_image = cv2.addWeighted(screen,0.8, line_image,1.2,2)
        cv2.imshow('lane-detection',cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
        #autosteer END

      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break