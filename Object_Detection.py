#!/usr/bin/env python3

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import pyautogui
import msvcrt as m

from Tracking import *
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append("..")


#cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture('chute01/cam1.avi')
cap5 = cv2.VideoCapture('chute01/cam5.avi')
cap4 = cv2.VideoCapture('chute01/cam4.avi')
#cap = cv2.VideoCapture('768x576.avi')

cam1Pos = []
cam4Pos = []
cam5Pos = []
tracks = []
maxTrack = 0

#the colour values for the tracks, cyan for current and red for dropped in this frame.
colour1 = (255, 255, 0)
colour2 = (0, 0, 255)

#tensorflow demo
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
#tensorflow demo

#setting up the camera video values
im_width1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
im_height1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)

im_width5 = cap5.get(cv2.CAP_PROP_FRAME_WIDTH)
im_height5 = cap5.get(cv2.CAP_PROP_FRAME_HEIGHT)

im_width4 = cap4.get(cv2.CAP_PROP_FRAME_WIDTH)
im_height4 = cap4.get(cv2.CAP_PROP_FRAME_HEIGHT)

#gets the first frame from camera1 and gets the transformation matrix and the points of interest as an image
ret, img1 = cap1.read()
#points on orignal image
pts1 = np.float32([[70, 130],[405, 480],[222, 95],[570, 305]])
#points to transform to
pts2 = np.float32([[50,50],[50,850],[450,50],[450,850]])
#the transformation matrix from cam feed point to floor point
M1 = cv2.getPerspectiveTransform(pts1,pts2)
#the reverse of hte previous matrix
M1i = np.linalg.inv(M1)
#the floor image
dst1 = cv2.warpPerspective(img1,M1,(500,900))
#a copy for later use
copyDst1 = dst1.copy()

#gets the first frame from camera4 and gets the transformation matrix and the points of interest as an image
ret, img5 = cap5.read()
pts1 = np.float32([[280, 190],[125, 410],[425, 192],[460, 450]])
pts2 = np.float32([[50,50],[50,850],[450,50],[450,850]])
M5 = cv2.getPerspectiveTransform(pts1,pts2)
M5i = np.linalg.inv(M5)
dst5 = cv2.warpPerspective(img5,M5,(500,900))

#gets the first frame from camera5 and gets the transformation matrix and the points of interest as an image
ret, img4 = cap4.read()
pts1 = np.float32([[487, 397],[395, 130],[130, 410],[250, 140]])
pts2 = np.float32([[50,50],[50,850],[450,50],[450,850]])
M4 = cv2.getPerspectiveTransform(pts1,pts2)
M4i = np.linalg.inv(M4)
dst4 = cv2.warpPerspective(img4,M4,(500,900))

#syncing video 5 to match video 1
#video 1 starts 20 frames in from the start of video 5
for x in range(0, 20):
    cap5.read()

#syncing video 4 to match video 5 and 1
#video 1 and 5 start 3 frames in from the start of video 4
for x in range(0, 3):
    cap4.read()

#setting the frame count
frame = 1

#to save as video later
#fps = cap1.get(cv2.CAP_PROP_FPS)
#set to 30 as the video's i am using are at 120 fps so when outputted like that can become hard to see the tracks and detection points
fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outCam1 = cv2.VideoWriter('outputCam1.avi', fourcc, fps, (int(im_width1), int(im_height1)))
outCam4 = cv2.VideoWriter('outputCam4.avi', fourcc, fps, (int(im_width4), int(im_height4)))
outCam5 = cv2.VideoWriter('outputCam5.avi', fourcc, fps, (int(im_width5), int(im_height5)))
outFloor = cv2.VideoWriter('outFloor.avi', fourcc, fps, (int(500), int(900)))

#skip to parts of the video so you dont need to wait as long for it to finish if only want a certain part done
#1250 for multi
#787 first walk in
for x in range(0,850):
    cap1.read()
    cap4.read()
    cap5.read()

#sets the font and size for the track number
try:
    font = ImageFont.truetype('arial.ttf', 24)
except IOError:
    font = ImageFont.load_default()

#makes the image usabele for PIL to wrtie the text
def toPil(image):
    return Image.fromarray(np.uint8(image)).convert('RGB')

#gets the values needed for the point tracking
def pointTransform(box):
    ymin, xmin, ymax, xmax = box
    xpos = xmax -(xmax-xmin)/2
    xPos = xpos * im_width1
    yPos = ymax * im_height1

    #into correct dimensions
    pts = np.array([[xPos,yPos]])
    return np.array([pts])

#writes the track number above the track point on each image
def printTrackText(tracks):
    write = []
    maxTrack = 0
    for i in range(len(tracks)):
          if i > maxTrack:  
            maxTrack = i
          write.append(i)
          #print(write)
          camPoints1 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M1i)
          camPoints4 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M4i)
          camPoints5 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M5i)

          if tracks[i][1][2] == 0:
              if tracks[i][1][0] == 0:
                  draw1.text((int(camPoints1[0][0][0] - 5.0) , int(camPoints1[0][0][1] - 30.0)), str(write[i]), colour1, font=font)
                  draw4.text((int(camPoints4[0][0][0] - 5.0) , int(camPoints4[0][0][1] - 30.0)), str(write[i]), colour1, font=font)
                  draw5.text((int(camPoints5[0][0][0] - 5.0) , int(camPoints5[0][0][1] - 30.0)), str(write[i]), colour1, font=font)
                  drawFloor.text((int(tracks[i][0][0][0] - 5.0) , int(tracks[i][0][0][1] - 30.0)), str(write[i]), colour1, font=font)
              
              elif tracks[i][1][0] == 1:
                  draw1.text((int(camPoints1[0][0][0] - 5.0) , int(camPoints1[0][0][1] - 30.0)), str(write[i]), colour2, font=font)
                  draw4.text((int(camPoints4[0][0][0] - 5.0) , int(camPoints4[0][0][1] - 30.0)), str(write[i]), colour2, font=font)
                  draw5.text((int(camPoints5[0][0][0] - 5.0) , int(camPoints5[0][0][1] - 30.0)), str(write[i]), colour2, font=font)
                  drawFloor.text((int(tracks[i][0][0][0] - 5.0) , int(tracks[i][0][0][1] - 30.0)), str(write[i]), colour2, font=font)
    print(maxTrack)
#draws the points and track of each current track
def imageFloorPoints(tracks):
      for i in range(len(tracks)):
          cam1Points = []
          cam4Points = []
          cam5Points = []
          
          if tracks[i][1][2] == 0:
              if tracks[i][1][0] == 0:
                  if len(tracks[i][0]) > 2:
                 
                      #draws a line on the image where the current tracks have been
                      for j in range((len(tracks[i][0])-1)):
                          cv2.line(dst1, (int(tracks[i][0][j][0]), int(tracks[i][0][j][1])),(int(tracks[i][0][j+1][0]), int(tracks[i][0][j+1][1])), colour1, 3)

                      for j in range(len(tracks[i][0])):
                          cam1Points.append(cv2.perspectiveTransform(np.array([[tracks[i][0][j]]]), M1i))
                          cam4Points.append(cv2.perspectiveTransform(np.array([[tracks[i][0][j]]]), M4i))
                          cam5Points.append(cv2.perspectiveTransform(np.array([[tracks[i][0][j]]]), M5i))

                      for j in range((len(cam1Points) - 1)):
                          cv2.line(image_np1, (int(cam1Points[j][0][0][0]),int(cam1Points[j][0][0][1])),(int(cam1Points[j+1][0][0][0]),int(cam1Points[j+1][0][0][1])), colour1, 3)
                          cv2.line(image_np4, (int(cam4Points[j][0][0][0]),int(cam4Points[j][0][0][1])),(int(cam4Points[j+1][0][0][0]),int(cam4Points[j+1][0][0][1])), colour1, 3)
                          cv2.line(image_np5, (int(cam5Points[j][0][0][0]),int(cam5Points[j][0][0][1])),(int(cam5Points[j+1][0][0][0]),int(cam5Points[j+1][0][0][1])), colour1, 3)
                          
                  #image 1
                  camPoints1 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M1i)
                  cv2.circle(image_np1, (int(camPoints1[0][0][0]), int(camPoints1[0][0][1])), 5, colour1, -1)
                  #image4
                  camPoints4 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M4i)
                  cv2.circle(image_np4, (int(camPoints4[0][0][0]), int(camPoints4[0][0][1])), 5, colour1, -1)
                  #image5
                  camPoints5 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M5i)
                  cv2.circle(image_np5, (int(camPoints5[0][0][0]), int(camPoints5[0][0][1])), 5, colour1, -1)
                  #floor
                  cv2.circle(dst1, (int(tracks[i][0][0][0]), int(tracks[i][0][0][1])), 5, colour1, -1)

              elif tracks[i][1][0] == 1:
                  if len(tracks[i][0]) > 2:
                  
                      #image1
                      camPoints1 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M1i)
                      cv2.circle(image_np1, (int(camPoints1[0][0][0]), int(camPoints1[0][0][1])), 5, colour2, -1)
                      #image4
                      camPoints4 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M4i)
                      cv2.circle(image_np4, (int(camPoints4[0][0][0]), int(camPoints4[0][0][1])), 5, colour2, -1)
                      #image5
                      camPoints5 = cv2.perspectiveTransform(np.array([[tracks[i][0][0]]]), M5i)
                      cv2.circle(image_np5, (int(camPoints5[0][0][0]), int(camPoints5[0][0][1])), 5, colour2, -1)
                      #floor
                      cv2.circle(dst1, (int(tracks[i][0][0][0]), int(tracks[i][0][0][1])), 5, colour2, -1)

#detection values
camera1Detections = 0
cam1total = 0
camera4Detections = 0
cam4total = 0
camera5Detections = 0
cam5total = 0
noDetection = 0

cam1correct = 0
cam4correct = 0
cam5correct = 0

actualLocationX = [104.4, 134.2, 161.7, 198.8, 248.9, 300.6, 347.7, 375.1, 378.4, 393.3, 393.8, 395.7, 393.4, 380.0, 370.8, 370.0, 342.2, 335.2, 318.2, 311.8, 298.7, 296.5, 270.4, 263.4]
actualLocationY = [-6.9, 32.5, 72.3, 119.7, 187.9, 229.4, 291.0, 341.3, 426.2, 474.5, 532.2, 581.4, 637.0, 686.0, 748.3, 752.3, 753.6, 758.1, 740.0, 732.9, 720.9, 715.7, 703.3, 704.2]

count=0
overall = 0

mouseClick = False

detectionThreshold = .65
mergeDistance = 70

#strating the main loop
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      
      if frame > 240:
          cv2.destroyAllWindows()
          break

      #to end the program if any of the feeds end
      ret, image_np1 = cap1.read()
      if not ret:
          cv2.destroyAllWindows()
          break

      ret, image_np5 = cap5.read()
      if not ret:
          cv2.destroyAllWindows()
          break

      ret, image_np4 = cap4.read()
      if not ret:
          cv2.destroyAllWindows()
          break

      #so each loop the floor is reset back to the orignal
      dst1 = copyDst1.copy()

      # tensorflow
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np1_expanded = np.expand_dims(image_np1, axis=0)
      image_np5_expanded = np.expand_dims(image_np5, axis=0)
      image_np4_expanded = np.expand_dims(image_np4, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      #tensor flow

      #resetting the values for this frame
      found = 0
      cam1Pos = []
      nPoints1 = []
      cam4Pos = []
      nPoints4 = []
      cam5Pos = []
      nPoints5 = []

      #gets the boxes, scores what object it is and the number of detections
      (boxes1, scores1, classes1, num_detections1) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np1_expanded})

      # Visualization of the results of a detection. Comment out if dont want detection boxes
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np1,
          np.squeeze(boxes1),
          np.squeeze(classes1).astype(np.int32),
          np.squeeze(scores1),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=detectionThreshold,
          line_thickness=8)

      boxes1 = np.squeeze(boxes1)
      classes1 = np.squeeze(classes1).astype(np.int32)
      scores1 = np.squeeze(scores1)
      
      #keeps track of the points if of type person, as that is what we care about tracking, also transforms all of the points to floor position
      for i in range(boxes1.shape[0]):
        if scores1 is None or scores1[i] > detectionThreshold:
            if classes1[i] in category_index.keys():
              class_name = category_index[classes1[i]]['name']

            if class_name == 'person':
                camera1Detections+=1
                cam1total+=1
                pts = pointTransform(tuple(boxes1[i].tolist()))
                nPoints1.append(cv2.perspectiveTransform(pts, M1))

      #gets the boxes, scores what object it is and the number of detections
      (boxes5, scores5, classes5, num_detections5) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np5_expanded})

      # Visualization of the results of a detection. Comment out if dont want detection boxes
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np5,
          np.squeeze(boxes5),
          np.squeeze(classes5).astype(np.int32),
          np.squeeze(scores5),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=detectionThreshold,
          line_thickness=8)

      boxes5 = np.squeeze(boxes5)
      classes5 = np.squeeze(classes5).astype(np.int32)
      scores5 = np.squeeze(scores5)

      #keeps track of the points if of type person, as that is what we care about tracking, also transforms all of the points to floor position
      for i in range(boxes5.shape[0]):
        if scores5 is None or scores5[i] > detectionThreshold:
            if classes5[i] in category_index.keys():
              class_name = category_index[classes5[i]]['name']

            if class_name == 'person':
                camera5Detections+=1
                cam5total+=1
                pts = pointTransform(tuple(boxes5[i].tolist()))
                nPoints5.append(cv2.perspectiveTransform(pts, M5))

      (boxes4, scores4, classes4, num_detections4) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np4_expanded})

      # Visualization of the results of a detection. Comment out if dont want detection boxes
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np4,
          np.squeeze(boxes4),
          np.squeeze(classes4).astype(np.int32),
          np.squeeze(scores4),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=detectionThreshold,
          line_thickness=8)

      boxes4 = np.squeeze(boxes4)
      classes4 = np.squeeze(classes4).astype(np.int32)
      scores4 = np.squeeze(scores4)

      #keeps track of the points if of type person, as that is what we care about tracking, also puts all of the points to floor position
      for i in range(boxes4.shape[0]):
        if scores4 is None or scores4[i] > detectionThreshold:
            if classes4[i] in category_index.keys():
              class_name = category_index[classes4[i]]['name']

            if class_name == 'person':
                camera4Detections+=1
                cam4total+=1
                pts = pointTransform(tuple(boxes4[i].tolist()))
                nPoints4.append(cv2.perspectiveTransform(pts, M4))
      
      #copies points as some need to change and keep 
      cam1Pos = nPoints1[:]
      cam5Pos = nPoints5[:]
      cam4Pos = nPoints4[:]

      distance = []      
      cam1PosCopy = cam1Pos[:]
      cam4PosCopy = cam4Pos[:]
      cam5PosCopy = cam5Pos[:]

      floorPos = []
      finalFloor = []
      #0 for false
      added1 = 0

      #if camera1 doesnt have any detections just look at other camera
      if(len(cam1PosCopy) > 0):
          #if next camera doesnt have any detecions look at other
          if(len(cam5PosCopy) > 0):
                  #for all detection find the eculidian distance, if below a certain value set the new position as the mid point, else add the detections separately
                  for i in range(len(cam1PosCopy)):
                      for j in range(len(cam5PosCopy)):
                          distance = np.linalg.norm(cam1PosCopy[i][0][0] - cam5PosCopy[j][0][0])
                          if distance < mergeDistance:
                              floorPos.append((cam1PosCopy[i][0][0] + cam5PosCopy[j][0][0])/2)
                              cam1PosCopy[i][0][0] = [1000000,1000000]
                              cam5PosCopy[j][0][0] = [1000000,1000000]
                              added1 = 1
                      if added1 != 1:
                          if cam1PosCopy[i][0][0][0] < 100000:
                              floorPos.append(cam1PosCopy[i][0][0])
                      added1 = 0

                  for j in range(len(cam5PosCopy)):
                      if cam5PosCopy[j][0][0][0] < 100000 :
                          floorPos.append(cam5PosCopy[j][0][0])
          else:
              #as past the first if definitely detections so just add all detections
              for i in range(len(cam1Pos)):
                  floorPos.append(cam1Pos[i][0][0])
      else:
          #need to see if camera has detections, if so add all points as no points to match 
          if(len(cam5PosCopy) > 0):
              for i in range(len(cam5Pos)):
                  floorPos.append(cam5Pos[i][0][0])
      
      floorPosCopy = []
      floorPosCopy = floorPos[:]

      if(len(floorPosCopy) > 0):
          if(len(cam4PosCopy) > 0):
                  for i in range(len(floorPosCopy)):
                      for j in range(len(cam4PosCopy)):
                          #print(floorPosCopy[i])
                          distance = np.linalg.norm(floorPosCopy[i] - cam4PosCopy[j][0][0])
                          if distance < mergeDistance:
                              #middle point
                              newPos = (floorPosCopy[i] + cam4PosCopy[j][0][0])/2
                              finalFloor.append(newPos)
                              floorPosCopy[i] = [1000000,1000000]
                              cam4PosCopy[j][0][0] = [1000000,1000000]
                              added1 = 1
                          #else:
                              #floorPos.append(cam1PosCopy)
                              #if(cam4PosCopy[j][0][0][0] < 1000000):
                                  #finalFloor.append(cam4PosCopy[j][0][0])
                      if added1 != 1:
                          if(floorPosCopy[i][0] < 100000):
                              finalFloor.append(floorPosCopy[i])
                      added1 = 0

                  for j in range(len(cam4PosCopy)):
                      if(cam4PosCopy[j][0][0][0] < 100000):
                                  finalFloor.append(cam4PosCopy[j][0][0])
          else:
              for i in range(len(floorPos)):
                  finalFloor.append(floorPos[i])
      else:
          if(len(cam4Pos) > 0):
              for i in range(len(cam4Pos)):
                  finalFloor.append(cam4Pos[i][0][0])
      
      #print()
      #print()
     
      #print(np.array(floorPos).shape)
      #print(len(finalFloor))
      #print()
      #print(finalFloor)

      x = 0

      if len(finalFloor) > 0:
          while x < len(finalFloor):
              #print(finalFloor[x])
              if finalFloor[x][0] < 10000:
                  x += 1
              else:
                  del finalFloor[x]
      
      floorPoints = []
      #makes a array of all points tracked that have tried to be tied to each other
      floorPoints.append(np.array(finalFloor))
 

      #starting the tracking
      if len(floorPoints) > 0:
          tracks = objects(floorPoints[len(floorPoints)-1], tracks)
      
      #puts the tracks and points onto the iamges
      imageFloorPoints(tracks)

      #transform the images to be able to put text onto them
      image_pil1 = toPil(image_np1)
      image_pil4 = toPil(image_np4)
      image_pil5 = toPil(image_np5)
      image_pilFloor = toPil(dst1)
      
      draw1 = ImageDraw.Draw(image_pil1)
      draw4 = ImageDraw.Draw(image_pil4)
      draw5 = ImageDraw.Draw(image_pil5)
      drawFloor = ImageDraw.Draw(image_pilFloor)

      #puts the track value above the point
      printTrackText(tracks)

      #transform the images back
      image_np1 = np.array(image_pil1)
      image_np4 = np.array(image_pil4)
      image_np5 = np.array(image_pil5)
      dst1 = np.array(image_pilFloor)

      #if you want to see the video frame by frame as it is working
      #cv2.imshow('Cam 1', image_np1)
      #cv2.imshow('Cam 4', image_np4)
      #cv2.imshow('Cam 5', image_np5)
      #cv2.imshow('Floor', dst1)
      x1 = 0.0
      y1 = 0.0
      x2 = 0.0
      y2 = 0.0
      x3 = 0.0
      y3 = 0.0

      print(frame%10)
      if(frame%10 == 0):
          
          #cv2.imshow('Cam 1', image_np1)
          #m.getch()
          #x1, y1 = pyautogui.position()
          #cv2.imshow('Cam 4', image_np4)
          #m.getch()
          #x2, y2 = pyautogui.position()
          #cv2.imshow('Cam 5', image_np5)
          #m.getch()
          #x3, y3 = pyautogui.position()
          
          ##print(x1, '   ', y1)
          ##print(x2, '   ', y2)
          ##print(x3, '   ', y3)

          #x1 = x1 - 29
          #x2 = x2 - 633
          #x3 = x3 - 752
          #y1 = y1 - 503
          #y2 = y2 - 32
          #y3 = y3 - 502

          #pts1 = np.array([[x1,y1]])
          #pts1 = np.array([pts1])
          #pts2 = np.array([[x2,y2]])
          #pts2 = np.array([pts2])
          #pts3 = np.array([[x3,y3]])
          #pts3 = np.array([pts3])
          ##print(pts1.astype(float))

          #cam1point = cv2.perspectiveTransform(pts1.astype(float), M1)
          #cam4point = cv2.perspectiveTransform(pts2.astype(float), M4)
          #cam5point = cv2.perspectiveTransform(pts3.astype(float), M5)

          #overalPoint = (cam1point+cam4point)/2
          overalPoint = (actualLocationX[count],actualLocationY[count])

          print(overalPoint)

          dist = 1000000
          closest = 0

          for i in range(len(floorPoints[0])):
            if(np.linalg.norm(floorPoints[0][i] - overalPoint) < dist):
                dist = np.linalg.norm(floorPoints[0][i] - (overalPoint))
                closest = i
          
          if dist > 300:
              dist = 0

          print(dist)
          count+=1
          overall += dist
      else:
          cv2.imshow('Cam 1', image_np1)
          cv2.imshow('Cam 4', image_np4)
          cv2.imshow('Cam 5', image_np5)

      #if x1 != 0:

      #print(len(floorPoints[0]))
      #print(floorPoints)


      #output the images into a video
      outCam1.write(image_np1)
      outCam4.write(image_np4)
      outCam5.write(image_np5)
      outFloor.write(dst1)
      
      #print(frame)
      #print()
      #print('---------')

      if camera1Detections == 1:
          cam1correct +=1
      if camera4Detections == 1:
          cam4correct +=1
      if camera5Detections == 1:
          cam5correct+=1

      if camera1Detections > 1:
        cam1correct += 1
      if camera4Detections > 1:
        cam4correct +=1
      if camera5Detections > 1:
        cam5correct+=1

      
      if ((camera1Detections == 0) & (camera4Detections == 0) & (camera5Detections == 0)):
          noDetection+=1

      camera1Detections = 0
      camera4Detections = 0
      camera5Detections = 0
      frame += 1

      print('---')
      #print(cam1correct)
      #print(cam1total)
      #print()
      #print(cam4correct)
      #print(cam4total)
      #print()
      #print(cam5correct)
      #print(cam5total)
      #print()
      #print(noDetection)

      #print(frame)

      #press q to end the program
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
      #print(len(tracks))
      #print(maxTrack)

print('Average distance = ', overall/count)

cap1.release
cap4.release
cap5.release
cv2.destroyAllWindows()