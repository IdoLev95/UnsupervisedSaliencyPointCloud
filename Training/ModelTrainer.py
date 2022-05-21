import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm import tqdm
import sys
sys.path.append('/content/UnsupervisedSaliencyPointCloud/Models')
from model import PointNetCls, feature_transform_regularizer


class ModelPointNetTrainer():
  def __init__(self, train_data_loader, test_data_loader,classifier,optimizer,scheduler,batch_size):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.batch_size = batch_size
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler

  def train(self,num_epochs,feature_transform = False):
    for epoch in range(num_epochs):
      with tqdm(self.train_data_loader, unit="batch") as tepoch:
        lossOfEpoch = []
        accOfEpoch = []
        self.scheduler.step()
        for batchInd,data in enumerate(tepoch,0):
          tepoch.set_description(f"Epoch {epoch}")
          points, target = data
          target = target[:, 0]
          points = points.transpose(2, 1)
          points, target = points.cuda(), target.cuda()
          self.optimizer.zero_grad()
          pred, trans, trans_feat,_ = self.classifier(points)  
          loss = F.nll_loss(pred, target)
          if feature_transform:
              loss += feature_transform_regularizer(trans_feat) * 0.001
          loss.backward()
          self.optimizer.step()
          pred_choice = pred.data.max(1)[1]
          correct = pred_choice.eq(target.data).cpu().sum()
          if batchInd % 10 == 0:
              j, data = next(enumerate(self.test_data_loader, 0))
              points, target = data
              target = target[:, 0]
              points = points.transpose(2, 1)
              points, target = points.cuda(), target.cuda()
              classifier = classifier.eval()
              pred, _, _ ,_= classifier(points)
              loss = F.nll_loss(pred, target)
              pred_choice = pred.data.max(1)[1]
              correct = pred_choice.eq(target.data).cpu().sum()
              lossOfEpoch.append(loss.item())
              accOfEpoch.append(correct.item() / float(self.batchSize))
              tepoch.set_postfix(loss=sum(lossOfEpoch)/len(lossOfEpoch), accuracy=100. * sum(accOfEpoch)/len(accOfEpoch))
          

      torch.save(classifier.state_dict(), '/content/UnsupervisedSaliencyPointCloud/cls/cls_model_%d.pth' % (epoch))

        

