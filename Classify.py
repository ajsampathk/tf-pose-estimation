import torch
from torch.autograd import Variable 
from ModelClass import LinearModel
import rospy
from std_msgs.msg import Float64MultiArray

rospy.init_node("Classifier")
labels = ['hello','hands','namesthe','sitting','standing']

print("Loading Model..")
model = LinearModel(len(labels))
print(model.load_state_dict(torch.load("trained_classifier_net.pth")))
model.eval()

def cb(msg):
  angles = Variable(torch.Tensor([msg.data]))
  pred = model(angles)
  pred_label = labels[pred.data[0].tolist().index(max(pred.data[0].tolist()))]
  rospy.loginfo("{}-{}".format(pred_label,pred.data))

rospy.Subscriber('/poseangles',Float64MultiArray,cb)
while not rospy.is_shutdown():
  rospy.spin()
  
  
