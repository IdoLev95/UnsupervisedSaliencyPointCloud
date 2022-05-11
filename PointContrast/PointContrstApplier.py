# import
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm import tqdm
import numpy as np
import sys
  
# adding Folder_2 to the system path
sys.path.append('/content/UnsupervisedSaliencyPointCloud/Dataset')
from DataSet import ShapeNetDataset
sys.path.append('/content/UnsupervisedSaliencyPointCloud/Models')

from model import PointNetCls, feature_transform_regularizer
  

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--numPointsInEmbbeding', type=int, default = 50, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
# Arrange The data
opt = parser.parse_args()
print(opt)
blue = lambda x: '\033[94m' + x + '\033[0m'
if opt.dataset_type == 'shapenet':
    opt.dataset = '/content/drive/MyDrive/shapenetcore_partanno_segmentation_benchmark_v0'
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)
classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
def GetRandomSymmetricMat(N):
  x1 = np.zeros((1,9)).squeeze(0)
  x1[0:6] = np.random.uniform(0,2, 6)
  x1[7] = x1[5]
  x1[6] = x1[2]
  x1[8] = x1[3]
  x1[3] = x1[1]
  x1= torch.tensor(x1.reshape(3,3)).unsqueeze(0)
  x1 = x1.repeat(N,1,1)

  return x1
  
def ApplyAugOnPoints(points,numberOfPoints, maxPToDrop = 0.3):
  # Random drop points
  pDrop = np.random.uniform(0,maxPToDrop, 1)
  indicesToKeep = torch.randperm(numberOfPoints)
  indicesToKeep = indicesToKeep[(int)(pDrop * numberOfPoints) : ]
  augPoints = points[:,indicesToKeep,:]
  # Random rotate on points
  batchSize = points.shape[0]
  rotationMat = GetRandomSymmetricMat(batchSize)
  augPoints = torch.bmm(augPoints,rotationMat)
  # Concat tensors
  return torch.cat([points , augPoints] , dim = 0)
  
def CalcLossFromEmbbeding(embedding):
  return -1
num_batch = len(dataset) / opt.batchSize
classifier = classifier.train()
for epoch in range(opt.nepoch):
   with tqdm(dataloader, unit="batch") as tepoch:
      
      scheduler.step()
      for batchInd,data in enumerate(tepoch,0):
          tepoch.set_description(f"Epoch {epoch}")
          points, target = data
          target = target[:, 0]
          points = points.transpose(2, 1)
          points, target = points.cuda(), target.cuda()
          optimizer.zero_grad()
          # Apply Random Augmentaion on points
          AugPoints = ApplyAugOnPoints(points)
          _,_,_,embedding = classifier(AugPoints)
          
          loss = CalcLossFromEmbbeding(embedding)
          

          loss.backward()
          optimizer.step()