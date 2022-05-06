import argparse
import numpy as np
import torch
import socket
import argparse
import importlib
import copy
import os
import sys
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm import tqdm
sys.path.append('/content/UnsupervisedSaliencyPointCloud/Models')
from model import PointNetCls, feature_transform_regularizer
 
# adding Folder_2 to the system path
sys.path.append('/content/UnsupervisedSaliencyPointCloud/Dataset')
from DataSet import ShapeNetDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--Th', type=int, default=1, help='input threshold')
parser.add_argument(
    '--label', type=int, default=1, help='input label')
parser.add_argument(
    '--num_classes', type=int, default=16, help='classifier')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--modelPath',type = str, default = '/content/UnsupervisedSaliencyPointCloud/cls/cls_model_0.pth', help="use feature transform")
opt = parser.parse_args()
classifier = PointNetCls(k=opt.num_classes, feature_transform=False)
classifier.load_state_dict(torch.load(opt.modelPath))

classifier.cuda()
classifier.feat.activateBackwrdHook()

num_points = 2500
datasetPath = '/content/drive/MyDrive/shapenetcore_partanno_segmentation_benchmark_v0'
dataset = ShapeNetDataset(
        root=datasetPath,
        classification=True,
        npoints=num_points)
batchSize = 32
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers=int(2))



print(opt)
classifier.eval()
#wantedSignal = torch.zeros([opt.num_classes, 1])
#wantedSignal[opt.label] = 1

with tqdm(dataloader, unit="batch") as tepoch:

  for batchInd,data in enumerate(tepoch,0):
      
      points, target = data
      batchSize = points.shape[0]
      target = target[:, 0]
      points = points.transpose(2, 1)
      points, target = points.cuda(), target.cuda()
      
      classifier.feat.activateBackwrdHook()
      for indBatch in range(batchSize):
        
        pointCloud = points[indBatch,:,:].unsqueeze(0)
        singleTarget = target[indBatch].unsqueeze(0)
        if singleTarget != opt.label:
          continue
        pred = singleTarget
        while pred == singleTarget:
          pred, transwhole, trans_feat = classifier(pointCloud)
          #loss = F.nll_loss(pred, singleTarget)
          print(pred.shape)
          yc = pred[0,opt.label]
          yc.backward()
          gradientsPerPoint =torch.max(classifier.feat.gradients[0][0], 1, keepdim=False)[0]
          #### getA is the sum of features where gradientsPerPoint are the re the gradients according to the last layer
          getA = classifier(pointCloud, True)
          wholeFeatures = gradientsPerPoint* getA
          pred = torch.argmax(pred)
          newPointCloud = removeLargestInfluence(pointCloud,wholeFeatures)
        print(idle)
def removeLargestInfluence(pointCloud,wholeFeatures):
  
        