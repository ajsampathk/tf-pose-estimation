#! /usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import numpy as np

rospy.init_node('Tap')

def cb(msg):
  img = np.array(list(bytearray(msg.data)),dtype='uint8')
  rospy.loginfo(len(img))

rospy.Subscriber('/video/image_raw',Image,cb)

while not rospy.is_shutdown():
  rospy.spin()
