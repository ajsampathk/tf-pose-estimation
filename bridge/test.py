#! /usr/bin/env python

import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('Inferece_Bridge')

image_pub = rospy.Publisher('/Inference/image_raw',Image,queue_size=10)
bridge = CvBridge()

def cb(msg):
  img = np.array(list(bytearray(msg.data)),dtype='uint8')
  img = img.reshape(msg.width,msg.height,3)
  try:
    image_pub.publish(bridge.cv2_to_imgmsg(img,"bgr8"))
    rospy.loginfo("Inference image published")
  except CvBridgeError as e:
    print(e)


while not rospy.is_shutdown():
  rospy.Subscriber('/Pose/output/image_pre',Image,cb)
  rospy.spin()
