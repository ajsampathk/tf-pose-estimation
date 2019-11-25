#! /usr/bin/env python

import cv2
import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image


rospy.init_node('Video_Publisher')
image_pub = rospy.Publisher('/video/image_raw',Image,queue_size=10)

vid  = cv2.VideoCapture("./etcs/dance.mp4")

count = 0
state = 1

while state:
  image_out = Image()
  state, img = vid.read()
  try:
    (image_out.width,image_out.height,image_out.step) = img.shape
    image_out.encoding = "bgr8"
    image_out.is_bigendian = 0
    image_out.data = img.ravel().tolist()
    image_pub.publish(image_out)
    rospy.loginfo("Frame {} published".format(count))
  except Exception as e:
    rospy.logerr("Error publishing frame:{}".format(e))
 
  count += 1
