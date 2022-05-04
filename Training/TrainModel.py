from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
#from Dataset.DataSet import ShapeNetDataset
#from Models import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm import tqdm
# importing sys
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
#parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

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
#elif opt.dataset_type == 'modelnet40':
##    dataset = ModelNetDataset(
 #       root=opt.dataset,
 #       npoints=opt.num_points,
        #split='trainval')

  #  test_dataset = ModelNetDataset(
  #      root=opt.dataset,
  #      split='test',
  #      npoints=opt.num_points,
  #      data_augmentation=False)
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


num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
   with tqdm(dataloader, unit="batch") as tepoch:
      lossOfEpoch = []
      accOfEpoch = []
      
      scheduler.step()
      for batchInd,data in enumerate(tepoch,0):
          if(batchInd> 20):
            continue
          tepoch.set_description(f"Epoch {epoch}")
          points, target = data
          target = target[:, 0]
          points = points.transpose(2, 1)
          points, target = points.cuda(), target.cuda()
          optimizer.zero_grad()
          classifier = classifier.train()
          classifier.feat.activateBackwrdHook()
          pred, trans, trans_feat = classifier(points)
          
          loss = F.nll_loss(pred, target)
          
          if opt.feature_transform:
              loss += feature_transform_regularizer(trans_feat) * 0.001
          loss.backward()

          gradientsPerPoint =torch.max(classifier.feat.gradients[0][0], 1, keepdim=False)[0]
          #### getA is the sum of features where gradientsPerPoint are the re the gradients according to the last layer
          getA = classifier(points, True)
          #
          optimizer.step()
          pred_choice = pred.data.max(1)[1]
          correct = pred_choice.eq(target.data).cpu().sum()
          lossOfEpoch.append(loss.item())
          accOfEpoch.append(correct.item() / float(opt.batchSize))
          #print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
          tepoch.set_postfix(loss=sum(lossOfEpoch)/len(lossOfEpoch), accuracy=100. * sum(accOfEpoch)/len(accOfEpoch))
          if 1 % 10 == 0:
              j, data = next(enumerate(testdataloader, 0))
              points, target = data
              target = target[:, 0]
              points = points.transpose(2, 1)
              points, target = points.cuda(), target.cuda()
              classifier = classifier.eval()
              pred, _, _ = classifier(points)
              loss = F.nll_loss(pred, target)
              pred_choice = pred.data.max(1)[1]
              correct = pred_choice.eq(target.data).cpu().sum()
           #   print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
          

   torch.save(classifier.state_dict(), '/content/UnsupervisedSaliencyPointCloud/cls/cls_model_%d.pth' % (epoch))