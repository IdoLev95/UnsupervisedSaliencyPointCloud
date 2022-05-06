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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--Th', type=int, default=1, help='input threshold')
parser.add_argument(
    '--label', type=int, default=1, help='input label')
parser.add_argument(
    '--model', help='classifier')

parser.add_argument('--dataloader', help="dataloader")
opt = parser.parse_args()
print(opt)
class AdversialPointCloud():
    def __init__(self,Th):
        self.Th = Th
    def GetSaliencyGradCam(self, classifier,dataloader,label):
        with tqdm(dataloader, unit="batch") as tepoch:
      
          for batchInd,data in enumerate(tepoch,0):
              
              points, target = data
              batchSize = points.shape[0]
              target = target[:, 0]
              points = points.transpose(2, 1)
              points, target = points.cuda(), target.cuda()

              classifier.feat.activateBackwrdHook()
              for indBatch in range(batchSize):
                pointCloud = points[indBatch,:].unsqueeze(0)
                singleTarget = target.unsqueeze(0)
                pred, trans, trans_feat = classifier(pointCloud)
                loss = F.nll_loss(pred, singleTarget)
                loss.backward()
                gradientsPerPoint =torch.max(classifier.feat.gradients[0][0], 1, keepdim=False)[0]
                #### getA is the sum of features where gradientsPerPoint are the re the gradients according to the last layer
                getA = classifier(points, True)     