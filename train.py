import torch
from torch.autograd import Variable
import json
import sys
from ModelClass import LinearModel
from torchviz import make_dot

device = torch.device("cuda")

class PrepareDataset():
  
  labels = json.load(open("training_labels.json"))
  #labels = ['hello','hands','namesthe','sitting','stand']
  joints = ['N_RS','N_LS','RS_RE','LS_LE','RE_RW','LE_LW','N_RH','N_LH','RH_RK','LH_LK','RK_RA','LK_LA']
  dataset = []

  def __init__(self,ds):
    print("Preparing size ",len(ds)," Dataset")
    self.X = Variable(torch.Tensor([[data['angles'][joint] for joint in self.joints] for data in ds]))
    self.Y = Variable(torch.Tensor([[data['label']==label for label in self.labels] for data in ds]))

  def getVars(self):
    return self.X,self.Y


if __name__ =='__main__':
  ds = json.load(open("training_angles.json"))
  Dataset = PrepareDataset(ds)
  x,y = Dataset.getVars()
#print(x,y)

  model = LinearModel(len(Dataset.labels))

  criterion = torch.nn.MSELoss(reduction='mean')
  optimizer = torch.optim.SGD(model.parameters(),lr=0.07)

#print(model(x[0]))

  for epoch in range(100000):
    pred_y = model(x)
    loss = criterion(pred_y,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch {}, Loss {}".format(epoch,loss.data),end='\r')
    sys.stdout.flush()
  x = int(input("\nEnter index of element:"))
  new = Dataset.X[x]
  pred = model(new)
  pred_label = Dataset.labels[pred.data.tolist().index(max(pred.data.tolist()))]
  print("Prediction:",pred_label,pred.data)
  make_dot(pred.mean(),params = dict(model.named_parameters()))
  torch.save(model.state_dict(),"trained_classifier_net.pth")
