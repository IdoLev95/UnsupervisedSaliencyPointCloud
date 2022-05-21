import os
import torch.nn.functional as F
from torch.utils.data import dataloader
from tqdm import tqdm
import torch

class SaliencyDetectionUsingLabels():
  def __init__(self, classifier, train_data_loader, test_data_loader,num_points,Th = 0.99):
        self.classifier = classifier
        self.test_data_loader = test_data_loader
        self.train_data_loader = train_data_loader
        self.num_points = num_points
        self.Th =Th
  def removeLargestInfluence(self,pointCloud,wholeFeatures,pointToColorDict):
    sortedVal, indices = torch.sort(wholeFeatures)
    indices = indices.squeeze(0)
    remainInd = indices[0: (int)(len(indices) * self.Th)]
    indToColor = indices[(int)(len(indices) * self.Th) :]
    pointCloudNew =  pointCloud[: , :, remainInd]
    sortedVal = sortedVal.squeeze(0)
    #print(sortedVal.shape)

    maxT = sortedVal[-1]
    #print(maxT)
    normelizedT = wholeFeatures/maxT
    numPointsToColor = len(indToColor)
    for ind in range(numPointsToColor):
        currInd = indToColor[ind]
        currPointLoc = pointCloud[:,:,currInd].squeeze(0)
        currPointRed = normelizedT[0,currInd]
        currPointGreen = 1 - normelizedT[0,currInd]
        if(currPointLoc in pointToColorDict.keys()):
          print('you entered twice the same ind but how?')
        pointToColorDict[currPointLoc] = torch.tensor([currPointRed,currPointGreen,0])
    #print(pointToColorDict.values())
    return pointCloudNew , pointToColorDict

  def DetectSalientRegions(self):
    numPossibleCtegories = torch.zeros(self.num_points)
    self.classifier.eval()
    with tqdm(self.train_data_loader, unit="batch") as tepoch:
      for batchInd,data in enumerate(tepoch,0):
        points, target = data
        batchSize = points.shape[0]
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        
        self.classifier.feat.activateBackwrdHook()
        for indBatch in range(batchSize):
          
          pointCloud = points[indBatch,:,:].unsqueeze(0)
          singleTarget = target[indBatch].unsqueeze(0)
          numPossibleCtegories[singleTarget] = numPossibleCtegories[singleTarget]  + 1

          pred = singleTarget
          pointToColorDict = {}
          while pred == singleTarget:
            pred, transwhole, trans_feat,_ = self.classifier(pointCloud)

            pred = F.softmax(pred)
            yc = pred[0,singleTarget]
            yc.backward()
            gradientsPerPoint =torch.max(self.classifier.feat.gradients[0][0], 1, keepdim=False)[0]
            self.classifier.feat.gradients = []

            getA = self.classifier(pointCloud, True)
            wholeFeatures = gradientsPerPoint* getA
            pred = torch.argmax(pred)
            newPointCloud,pointToColorDict = self.removeLargestInfluence(pointCloud,wholeFeatures,pointToColorDict)
            print(str(yc.item()) + " " + str(pointCloud.shape[2] - newPointCloud.shape[2]) + 'points were removed and ' + str(newPointCloud.shape[2]) +  ' remained')
            pointCloud = newPointCloud
          
          remaindNumPoints = pointCloud.shape[2] 
          for ind in range(remaindNumPoints):
            #print(ind)
            #print(pointCloud.shape)
            currPointLoc = pointCloud[:,:,ind].squeeze(0)
            if(currPointLoc in pointToColorDict.keys()):
              print('you entered twice the same ind but how?')
            pointToColorDict[currPointLoc] = torch.tensor([0,0,0])
          # create json object from dictionary
          locAndColorArray = torch.zeros(self.num_points , 6)
          loc = list(pointToColorDict.keys())
          color = list(pointToColorDict.values())
          for ind in range(self.num_points):
            locAndColorArray[ind,0:3] = loc[ind]
            locAndColorArray[ind,3:] = color[ind]

          path =  '../../drive/MyDrive/saveRes/tensor' + str(int(singleTarget.item())) + '/'
          if(not os.path.exists(path)):
            os.makedirs(path)
          torch.save(locAndColorArray,path + str(int(numPossibleCtegories[singleTarget].item())) + '.pt')
          