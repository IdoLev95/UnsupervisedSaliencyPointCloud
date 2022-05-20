

class ModelTrainer():
  def __init__(self, train_data_loader, test_data_loader,classifier,optimizer,scheduler):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.classifier = classifier
        self.optimizer = optimizer
        self.scheduler = scheduler

  def train(num_epochs):
    for epoch in range(opt.nepoch):
      with tqdm(dataloader, unit="batch") as tepoch:
        lossOfEpoch = []
        accOfEpoch = []
        scheduler.step()
        for batchInd,data in enumerate(tepoch,0):
          tepoch.set_description(f"Epoch {epoch}")
          points, target = data
          target = target[:, 0]
          points = points.transpose(2, 1)
          points, target = points.cuda(), target.cuda()
          optimizer.zero_grad()
          pred, trans, trans_feat = classifier(points)  
          loss = F.nll_loss(pred, target)
          if opt.feature_transform:
              loss += feature_transform_regularizer(trans_feat) * 0.001
          loss.backward()
          optimizer.step()
          pred_choice = pred.data.max(1)[1]
          correct = pred_choice.eq(target.data).cpu().sum()
          if batchInd % 10 == 0:
              j, data = next(enumerate(testdataloader, 0))
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
              accOfEpoch.append(correct.item() / float(opt.batchSize))
              tepoch.set_postfix(loss=sum(lossOfEpoch)/len(lossOfEpoch), accuracy=100. * sum(accOfEpoch)/len(accOfEpoch))
          

   torch.save(classifier.state_dict(), '/content/UnsupervisedSaliencyPointCloud/cls/cls_model_%d.pth' % (epoch))

        

