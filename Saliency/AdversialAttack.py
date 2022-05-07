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
    '--Th', type=float, default=1, help='input threshold')
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
def removeLargestInfluence(pointCloud,wholeFeatures,Th,pointToColorDict):
    sortedVal, indices = torch.sort(wholeFeatures)
    indices = indices.squeeze(0)
    remainInd = indices[indices < (int)(len(indices) * Th)]
    indToColor = indices[indices >= (int)(len(indices) * Th)]
    pointCloudNew =  pointCloud[: , :, remainInd]
    maxT = sortedVal[-1]
    normelizedT = wholeFeatures/maxT
    numPointsToColor = len(indToColor)
    for ind in range(numPointsToColor):
        currInd = indToColor[ind]
        currPointLoc = pointCloud[:,:,currInd].squeeze(0)
        currPointRed = normelizedT[0,currInd]
        currPointGreen = 1 - normelizedT[0,currInd]
        if(currPointLoc in pointToColorDict.keys()):
          print('you entered twice the same ind but how?')
        pointToColorDict[currPointLoc] = np.array([currPointRed.detach().cpu(),currPointGreen.detach().cpu(),0])
    return pointCloudNew , pointToColorDict


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
        pointToColorDict = dict()
        while pred == singleTarget:
          pred, transwhole, trans_feat = classifier(pointCloud)
          #loss = F.nll_loss(pred, singleTarget)
          pred = F.softmax(pred)
          yc = pred[0,opt.label]
          yc.backward()
          gradientsPerPoint =torch.max(classifier.feat.gradients[0][0], 1, keepdim=False)[0]
          classifier.feat.gradients = []
          #### getA is the sum of features where gradientsPerPoint are the re the gradients according to the last layer
          getA = classifier(pointCloud, True)
          wholeFeatures = gradientsPerPoint* getA
          pred = torch.argmax(pred)
          newPointCloud,pointToColorDict = removeLargestInfluence(pointCloud,wholeFeatures,opt.Th,pointToColorDict)
          print(str(pointCloud.shape[2] - newPointCloud.shape[2]) + 'points were removed and ' + str(newPointCloud.shape[2]) +  ' remained')
          pointCloud = newPointCloud
        
        remaindNumPoints = pointCloud.shape[2] 
        for ind in range(num_points):
          print(ind)
          print(pointCloud.shape)
          currPointLoc = pointCloud[:,:,ind].squeeze(0)
          if(currPointLoc in pointToColorDict.keys()):
            print('you entered twice the same ind but how?')
          pointToColorDict[currPointLoc] = np.array([0,0,0])
        print(len(pointToColorDict.keys()))
        print(idle)

        