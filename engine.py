#! /usr/bin/env python3

print("Loading TF modules...This may take a few minutes")
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from sensor_msgs.msg import Image

print("Loading ROS modules..")
import rospy
from std_msgs.msg import String
import json

print("Starting ROS Node")
rospy.init_node('Pose_Inference')
human_pub = rospy.Publisher('/Pose/output/Humans',String,queue_size=10)
image_pub = rospy.Publisher('/Pose/output/image_pre',Image,queue_size=10)

print("Importing Model")
w,h = model_wh('368x368')
en = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size = (w,h))



def image_cb(msg):
  image_out = Image()
  img = np.array(list(bytearray(msg.data)),dtype='uint8')
  img = img.reshape(msg.width,msg.height,msg.step)
  humans = en.inference(img,resize_to_default=True,upsample_size = 4.0)
  infer = TfPoseEstimator.draw_humans(img,humans,imgcopy=True)
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

  (image_out.width,image_out.height,n) = infer.shape
  image_out.encoding = "bgr8"
  image_out.is_bigendian = 0
  image_out.data = infer.ravel().tolist()
  image_out.step = 3
  human_pub.publish(json.dumps(all_humans))
  image_pub.publish(image_out)


  rospy.loginfo("Inference complete. {} human succesfully detected".format(len(humans)))




if __name__=='__main__':
  ch=input("\n All Modules Ready..Start Inference?[Y/N]")

  if ch=='N' or ch=='n':
    print("Exiting..")
  else:
    while not rospy.is_shutdown():
      rospy.Subscriber('/video/image_raw',Image,image_cb)
      rospy.spin()
