#! /usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import json
import numpy as np
from math import atan

links = {'N_RS':('Neck','RShoulder'),'N_LS':('Neck','LShoulder'),'RS_RE':('RShoulder','RElbow'),'LS_LE':('LShoulder','LElbow'),'RE_RW':('RElbow','RWrist'),'LE_LW':('LElbow','LWrist'),'N_RH':('Neck','RHip'),'N_LH':('Neck','LHip'),'RH_RK':('RHip','RKnee'),'RK_RA':('RKnee','RAnkle'),'LH_LK':('LHip','LKnee'),'LK_LA':('LKnee','LAnkle')}

joints = ['N_RS','N_LS','RS_RE','LS_LE','RE_RW','LE_LW','N_RH','N_LH','RH_RK','LH_LK','RK_RA','LK_LA']

rospy.init_node('Angles')
angles_pub = rospy.Publisher('Inference/poseangles',Float64MultiArray,queue_size=10)

def jsoncb(msg):
  humans = json.loads(msg.data)
  angles = Float64MultiArray()

  links = {'N_RS':('Neck','RShoulder'),'N_LS':('Neck','LShoulder'),'RS_RE':('RShoulder','RElbow'),'LS_LE':('LShoulder','LElbow'),'RE_RW':    ('RElbow','RWrist'),'LE_LW':('LElbow','LWrist'),'N_RH':('Neck','RHip'),'N_LH':('Neck','LHip'),'RH_RK':('RHip','RKnee'),'RK_RA':('RKnee','RAnkle'),'LH_LK':('LHip','LKnee'),'LK_LA':('LKnee','LAnkle')}

  for human in humans:
    #rospy.loginfo(human)
    angle_data = {'N_RS':0,'N_LS':0,'RS_RE':0,'LS_LE':0,'RE_RW':0,'LE_LW':0,'N_RH':0,'N_LH':0,'RH_RK':0,'RK_RA':0,'LH_LK':0,'LK_LA':0}
    for joint in links:
      #rospy.loginfo(joint)
      try:
        angle_data[joint] = atan((human[0][links[joint][1]][1]-human[0][links[joint][0]][1])/(human[0][links[joint][1]][0]-human[0][links[joint][0]][0]))
       # rospy.loginfo(angle_data)
      except Exception as e:
        pass
    angles.data = [angle_data[joint] for joint in joints]
    rospy.loginfo(angles.data)
    angles_pub.publish(angles)
    

rospy.Subscriber('/Pose/output/Humans',String,jsoncb)

while not rospy.is_shutdown():
  rospy.spin()
