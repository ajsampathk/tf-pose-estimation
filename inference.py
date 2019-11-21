#! /usr/bin/env python3

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
print("Loading TF modules...This may take a few minutes")
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

print("Loading ROS modules..")
import rospy
from std_msgs.msg import String
import json

print("Starting ROS Node")
rospy.init_node('Pose_Inference')
human_pub = rospy.Publisher('/Pose/output/Humans',String,queue_size=10)
image_pub = rospy.Publisher('/Pose/output/image_pre',String,queue_size=10)

print("Importing Model")
w,h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size = (w,h))

ch=input("\n All Modules Ready..Type 'y' or 'Y' to Start Inference")

while ch=='y' or ch=='Y':


  image = cv2.imread('./images/apink1.jpg', cv2.IMREAD_COLOR)


  print("Starting Inference..")

  humans = e.inference(image,resize_to_default=True,upsample_size = 4.0)

  print("Drawing Pose")
  infer = TfPoseEstimator.draw_humans(image,humans,imgcopy=True)

  print("Publishing ROS Packet")
  #print((humans[0].body_parts[0]))
  all_humans = []
  for human in humans:
    _human = []
 
    for part in human.body_parts:
      human_joints = {}
      human_joints['joint'] = str(human.body_parts[part].get_part_name()).split('.')[1]
      human_joints['x'] = human.body_parts[part].x
      human_joints['y'] = human.body_parts[part].y
      human_joints['confidence'] = human.body_parts[part].score
      _human.append(human_joints)
    all_humans.append(_human)


  print("Total Humans detected: {}".format(len(all_humans)))
  human_pub.publish(json.dumps(all_humans))

  print(infer.shape)
  infer_list = infer.tolist()
  image_pub.publish(json.dumps(infer_list))
  ch=input("\n All Modules Ready..Type 'y' or 'Y' to Start Inference")
