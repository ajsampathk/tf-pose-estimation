#! /usr/bin/env python
print("Loading TF modules...This may take a few minutes")
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import json
from math import atan


print("Importing Model")
w,h = model_wh('256x256')
en = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size = (w,h))

import os

root_path = './images/Pose_DS'
labels = os.listdir(root_path)

print("\n\nFound {} Labels:".format(len(labels)))
print(labels)

images = {}

def getHumans(humans):
  all_humans = []
  print("Total Humans Detected:",len(humans))
  
  for human in humans:
    _human = []
 
    human_joints = {}
    print("No. of Joints:",len(human.body_parts))
    for part in human.body_parts:
      
      human_joints[str(human.body_parts[part].get_part_name()).split('.')[1]] = (human.body_parts[part].x,human.body_parts[part].y)

    _human.append(human_joints)
    all_humans.append(_human)   
  print("Length of op:",len(all_humans))
  return all_humans

for label in labels:
  images[label] = []
  path = os.path.join(root_path,label)
  count = 1
  for image in os.listdir(path):
    img = cv2.imread(os.path.join(path,image),cv2.IMREAD_COLOR)
    humans = en.inference(img,resize_to_default=True,upsample_size = 4.0) 
    print("Processing {} images for label: {}".format(count,label))
    humans = getHumans(humans)
    if(len(humans)==1):
      images[label].append(humans[0])
    
    count+=1


print("Successfully obtained point data for labels..")

#with open('Dataset.json','w') as jsonfile:
#  json.dump(images,jsonfile)    


Angles = []

links = {'N_RS':('Neck','RShoulder'),'N_LS':('Neck','LShoulder'),'RS_RE':('RShoulder','RElbow'),'LS_LE':('LShoulder','LElbow'),'RE_RW':('RElbow','RWrist'),'LE_LW':('LElbow','LWrist'),'N_RH':('Neck','RHip'),'N_LH':('Neck','LHip'),'RH_RK':('RHip','RKnee'),'RK_RA':('RKnee','RAnkle'),'LH_LK':('LHip','LKnee'),'LK_LA':('LKnee','LAnkle')}

print("Obtaining Angle data..")

for label in images:
  for humans in images[label]:
    print("Processing angle data for label[",label,"]")
    pose = {}
    pose['label'] = label
    angle_data = {'N_RS':0,'N_LS':0,'RS_RE':0,'LS_LE':0,'RE_RW':0,'LE_LW':0,'N_RH':0,'N_LH':0,'RH_RK':0,'RK_RA':0,'LH_LK':0,'LK_LA':0}

    for joint in links:
      
      print(joint,':',end='')
      try:
        angle_data[joint] = atan((humans[0][links[joint][1]][1]-humans[0][links[joint][0]][1])/(humans[0][links[joint][1]][0]-humans[0][links[joint][0]][0]))
        print(angle_data[joint],end=' ,')
      except Exception as e:
        print(e," Not found")
        pass
    
    pose['angles'] = angle_data
    Angles.append(pose)
    print("\n")

print("Writing training data to file 'training_angles.json'...")

with open('training_angles.json','w') as jsonfile:
  json.dump(Angles,jsonfile)  



