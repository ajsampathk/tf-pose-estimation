import torch
from torch.autograd import Variable 
from ModelClass import LinearModel
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import json

rospy.init_node("Classifier")
label_pub = rospy.Publisher("Inference/prediction",String,queue_size  =10)
labels = json.load(open("training_labels.json"))

print("Loading Model..")
model = LinearModel(len(labels))
print(model.load_state_dict(torch.load("trained_classifier_net.pth")))
model.eval()

def cb(msg):
  angles = Variable(torch.Tensor([msg.data]))
  pred = model(angles)
  pred_label = labels[pred.data[0].tolist().index(max(pred.data[0].tolist()))]
  rospy.loginfo("{}- confidence={}".format(pred_label,max(pred.data[0].tolist())))
  label_pub.publish(pred_label)

rospy.Subscriber('Inference/poseangles',Float64MultiArray,cb)
while not rospy.is_shutdown():
  rospy.spin()
  
  
