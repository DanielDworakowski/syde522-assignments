#!/usr/bin/env python
import torch
import DataUtil
import argparse
import dataloader
import numpy as np
import progressbar
from debug import *
import mininet as mn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dl
from torch.autograd import Variable
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--nSplit', dest='nSplit', default=10, type=int, help='How many splits to use in KFold cross validation.')
    parser.add_argument('--numEpochs', dest='numEpochs', default=32, type=int, help='How many splits to use in KFold cross validation.')
    parser.add_argument('--batchSize', dest='bSize', default=16, type=int, help='How many splits to use in KFold cross validation.')
    args = parser.parse_args()
    return args
#
# Main loop for running the agent.
def train(args, imgs, labels, img_val, label_val):
    # 
    # Create the image augmentation.
    t = transforms.Compose([
            DataUtil.ToPIL(),
            DataUtil.RandomFlips(),
            DataUtil.ToTensor(),
            DataUtil.Normalize([0.59008044], np.sqrt([0.06342617])),
        ])
    topil = transforms.ToPILImage()
    train = dataloader.npdataset(imgs, labels.view(-1), t)
    validation = dataloader.npdataset(img_val, label_val.view(-1), t)
    stages = {
        'train': torch.utils.data.DataLoader(train, batch_size=args.bSize, shuffle=True, num_workers=0),
        'val': torch.utils.data.DataLoader(validation, batch_size=args.bSize, shuffle=False, num_workers=0),
    }
    model = mn.Mininet()
    usegpu = torch.cuda.is_available()
    criteria = nn.CrossEntropyLoss()
    if usegpu:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    bestModel = model.state_dict()
    bestAcc = 0
    # 
    # Iterate.
    for epoch in range(args.numEpochs):
        printColour('Epoch {}/{}'.format(epoch, args.numEpochs - 1), colours.OKBLUE)
        for stage in stages:
            print('Stage: ', stage)
            #
            # Switch on / off gradients.
            model.train(stage == 'train')
            #
            # The current loss.
            runningLoss = 0.0
            runningCorrect = 0.0
            loader = stages[stage]
            # 
            # Progress bar.
            numMini = len(stages[stage])
            pbar = progressbar.ProgressBar(max_value=numMini-1) 
            # 
            # Train.
            for i, data in enumerate(loader):
                inputs, labels_cpu = data['img'], data['label']
                if usegpu:
                    labels_cpu.squeeze_()
                    inputs, labels = Variable(inputs).cuda(async = True), Variable(labels_cpu).cuda(async = True)
                else:
                    inputs, labels = Variable(inputs), Variable(labels_cpu)
                # 
                # Forward through network.
                out = model(inputs)
                #
                # Backward pass.
                optimizer.zero_grad()
                _, preds = torch.max(out.data, 1)
                loss = criteria(out, labels)
                #
                #  Backwards pass.
                if stage == 'train':
                    loss.backward()
                    optimizer.step()
                dCorrect = torch.sum(preds == labels.data)
                #
                #  Stats.
                runningLoss += loss.data[0]
                runningCorrect += dCorrect
                pbar.update(i)
            #
            # Overall stats
            epochLoss = runningLoss / len(stages[stage])
            epochAcc = runningCorrect / (len(stages[stage]) * args.bSize)
            #
            # Check if we have the new best model.
            isBest = False
            if stage == 'val' and epochAcc > bestAcc:
                isBest = True
                bestAcc = epochAcc
                bestModel = model.state_dict()
            #
            # Print per epoch results.
            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(stage, epochLoss, epochAcc))
    printColour('Best validation performance:%d'%(bestAcc), colours.OKGREEN)
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    imgs = np.expand_dims(np.load('data/X.npy').astype(np.float32) / 255, 1)
    labels = np.load('data/Y.npy')
    # 
    # Image statistics. 
    curMean = np.mean(imgs, axis=(0,2,3))
    curVar = np.var(imgs, axis=(0,2,3))
    kf = KFold(n_splits=args.nSplit, shuffle = True)
    for train_index, test_index in kf.split(imgs): 
        X_train, X_test = torch.from_numpy(imgs[train_index]), torch.from_numpy(imgs[test_index])
        y_train, y_test = torch.from_numpy(labels[train_index]), torch.from_numpy(labels[test_index])
        train(args, X_train, y_train, X_test, y_test)