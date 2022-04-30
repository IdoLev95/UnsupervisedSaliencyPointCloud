import argparse
import numpy as np
import pytorch
import socket
import importlib
import copy
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class AdversialPointCloud():
    def __init__(self, BATCH_SIZE, NUM_CLASSES,Th):
        self.Th = Th
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_CLASSES = NUM_CLASSES
    def GetSaliencyGradCam(self, model,pointCloud,label):
        pred = np.argmax(model.forward(pointCloud))
        pointCloudCopy = pointCloud.Copy()

        while pred == label:
          derivatives = model.GetDevs(pointCloudCopy)
          weights = model.GetWeights(pointCloudCopy,derivatives)
          pointCloudCopy,placesWhereRemovedAndScores = removeNeededPoints(pointCloudCopy,weights,self.Th)
          pred = np.argmax(model.forward(pointCloud))
        return dictPlacesAndScores
        