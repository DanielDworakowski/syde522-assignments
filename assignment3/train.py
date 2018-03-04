#!/usr/bin/env python
import gc
import sys
import copy
import math
import torch
import DataUtil
import argparse
import dataloader
import numpy as np
import progressbar
from debug import *
import pandas as pd
import pjReddieNet as mn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dl
from time import gmtime, strftime
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--nSplit', dest='nSplit', default=6, type=int, help='How many splits to use in KFold cross validation.')
    parser.add_argument('--numEpochs', dest='numEpochs', default=128, type=int, help='How many splits to use in KFold cross validation.')
    parser.add_argument('--batchSize', dest='bSize', default=32, type=int, help='How many splits to use in KFold cross validation.')
    parser.add_argument('--useTB', dest='useTB', default=False, action='store_true', help='Whether or not to log to Tesnor board.')
    parser.add_argument('--extra', dest='extra', default=1, type=int,  help='Whether or not to log to Tesnor board.')
    parser.add_argument('--cropsize', dest='cropSize', default=160, type=int,  help='Whether or not to log to Tesnor board.')
    parser.add_argument('--ens', dest='useENS', default=False, action='store_true', help='Whether to use an enemble of networks.')
    args = parser.parse_args()
    return args
#
# Setup tensorboard as require.
def doNothing(logger = None, model = None, tmp = None):
    pass
#
# Run every epoch.
def logEpochTensorboard(logger, model, epochSummary):
    logger.add_scalar('%s_loss'%epochSummary['phase'], epochSummary['loss'], epochSummary['epoch'])
    logger.add_scalar('%s_acc'%epochSummary['phase'], epochSummary['acc'], epochSummary['epoch'])
    labels = epochSummary['data']['label']
    for i in range(epochSummary['data']['label'].shape[0]):
        logger.add_image('{}_image_i-{}_epoch-{}_pre-:{}_label-{}'.format(epochSummary['phase'], i, epochSummary['epoch'], epochSummary['pred'][i], int(labels[i])), epochSummary['data']['img'][i]*math.sqrt(0.06342617) + 0.59008044, epochSummary['epoch'])
    for name, param in model.named_parameters():
        logger.add_histogram(name, param.clone().cpu().data.numpy(), epochSummary['epoch'])
#
# Write everything as needed.
def closeTensorboard(logger):
    logger.close()
#
# Main loop for running the agent.
def train(args, imgs, labels, img_val, label_val, modelConst):
    #
    # Create the image augmentation.
    t = transforms.Compose([
            DataUtil.ToPIL(),
            DataUtil.RandomFlips(),
            # DataUtil.RandomRotation(5),
            # DataUtil.ColourJitter(0.1, 0.1, 0.1, 0),
            DataUtil.RandomResizedCrop(args.cropSize, (0.5, 1.3)),
            DataUtil.ToTensor(),
            DataUtil.Normalize([0.59008044], np.sqrt([0.06342617])),
            # DataUtil.TenCrop(140, [0.59008044], np.sqrt([0.06342617])),
            #
        ])
    t_test = transforms.Compose([
            DataUtil.ToPIL(),
            DataUtil.RandomFlips(),
            DataUtil.TenCrop(args.cropSize, [0.59008044], np.sqrt([0.06342617])),
        ])
    # RandomRotation
    # FiveCrop
    # RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)

    topil = transforms.ToPILImage()
    train = dataloader.npdataset(imgs, labels.view(-1), t)
    validation = dataloader.npdataset(img_val, label_val.view(-1), t_test)
    stages = {
        'train': torch.utils.data.DataLoader(train, batch_size=args.bSize, shuffle=True, num_workers=0, pin_memory = True),
        'val': torch.utils.data.DataLoader(validation, batch_size=args.bSize, shuffle=False, num_workers=0, pin_memory = True),
    }
    model = modelConst()
    usegpu = torch.cuda.is_available()
    criteria = nn.CrossEntropyLoss()
    #
    # Whether to use the GPU.
    if usegpu:
        model.cuda()
    #
    # Type of optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2)
    bestModel = copy.deepcopy(model.state_dict())
    bestAcc = 0
    logger = None
    logEpoch = doNothing
    closeLogger = doNothing
    if args.useTB:
        logger = SummaryWriter()
        logEpoch = logEpochTensorboard
        closeLogger = closeTensorboard
    #
    # Iterate.
    for epoch in range(args.numEpochs):
        printColour('Epoch {}/{}'.format(epoch, args.numEpochs - 1), colours.OKBLUE)
        for stage in stages:
            gc.collect()
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
            numMini = len(loader)
            pbar = progressbar.ProgressBar(max_value=numMini-1)
            #
            # Train.
            for i, data in enumerate(loader):
                inputs_cpu, labels_cpu = data['img'], data['label']
                if usegpu:
                    labels_cpu.squeeze_()
                    inputs, labels = Variable(inputs_cpu, requires_grad=False).cuda(async = True), Variable(labels_cpu, requires_grad=False).cuda(async = True)
                else:
                    inputs, labels = Variable(inputs_cpu, requires_grad=False, volatile=True), Variable(labels_cpu, requires_grad=False, volatile=True)
                #
                # Forward through network.
                if stage == 'train':
                    out = model(inputs)
                else:
                    #
                    # The 5 crop from above takes the corners of the iamge and center.
                    # We must now average the contributions.
                    bs, ncrops, c, h, w = inputs_cpu.size()
                    inputs = inputs.view(-1, c, h, w)
                    result = model(inputs) # fuse batch size and ncrops
                    out = result.view(bs, ncrops, -1).mean(1) # avg over crops
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
            #
            # Summary for logging in TB.
            summary = {
                'phase': stage,
                'epoch': epoch,
                'loss': epochLoss,
                'acc': epochAcc,
                'data': data,
                'pred' : preds
            }
            logEpoch(logger, model, summary)
    printColour('Best validation performance:%f'%(bestAcc), colours.OKGREEN)
    closeLogger(logger)
    retModel = copy.deepcopy(bestModel)
    for key, val in retModel.items():
        retModel[key] = val.cpu()
    return retModel, bestAcc
#
# Evaluate and return predictions.
def evalModel(args, imgs, model):
    t_test = transforms.Compose([
            DataUtil.ToPIL(),
            # DataUtil.RandomFlips(),
            DataUtil.TenCrop(args.cropSize, [0.59008044], np.sqrt([0.06342617])),
        ])
    labels = torch.zeros(imgs.shape[0])
    test = dataloader.npdataset(imgs, labels.view(-1), t_test)
    loader = torch.utils.data.DataLoader(test, batch_size=args.bSize, shuffle=False, num_workers=0)
    outs = []
    usegpu = torch.cuda.is_available()
    if usegpu:
        model.cuda()
    #
    # Iterate.
    for data in loader:
        gc.collect()
        inputs, labels_cpu = data['img'], data['label']
        if usegpu:
            labels_cpu.squeeze_()
            inputs, labels = Variable(inputs, requires_grad=False).cuda(async = True), Variable(labels_cpu, requires_grad=False).cuda(async = True)
        else:
            inputs, labels = Variable(inputs, requires_grad=False), Variable(labels_cpu, requires_grad=False)
        #
        # The 5 crop from above takes the corners of the iamge and center.
        # We must now average the contributions.
        bs, ncrops, c, h, w = inputs.size()
        result = model(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
        out = result.view(bs, ncrops, -1).mean(1) # avg over crops
        # print(outs)
        outs.append(copy.copy(out.cpu()))
    outs = torch.cat(outs, dim=0)
    return outs
#
# Iterate and get results of an ensemble.
def runEnsemble(mdlConst, modelsData, img_test):
    #
    # Use the ensemble.
    results = []
    X_test = torch.from_numpy(img_test)
    for mdl, acc in modelsData:
        model = mdlConst()
        model.load_state_dict(mdl)
        model.cpu()
        model.train(False)
        result = evalModel(args, X_test, model)
        results.append(result)
    #
    # Begin averaging.
    allResults = torch.stack(results, dim=1)
    #
    # Average.
    # out = allResults.mean(1)
    # _, preds = torch.max(out.data, 1)
    #
    # Vote.
    # out = allResults.argmax(2)
    energy, classVotes = allResults.max(2)
    votes = np.zeros((img_test.shape[0], 20))
    #
    # There is likely a fancy numpy way to do this.
    for i, vote in enumerate(classVotes):
        for voteIdx in vote.data.numpy():
            votes[i, voteIdx] += 1
    #
    # Count up the votes.
    preds = torch.Tensor(np.argmax(votes, axis=1))
    return preds
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    imgs = np.expand_dims(np.load('data/X.npy').astype(np.float32) / 255, 1)
    labels = np.load('data/Y.npy')
    img_test =  np.expand_dims(np.load('data/X_test.npy').astype(np.float32) / 255, 1)
    #
    # Image statistics.
    # curMean = np.mean(imgs, axis=(0,2,3))
    # curVar = np.var(imgs, axis=(0,2,3))
    models = []
    for i in range(args.extra):
        kf = KFold(n_splits=args.nSplit, shuffle = True)
        mdlConst = mn.Mininet
        for train_index, test_index in kf.split(imgs):
            X_train, X_test = torch.from_numpy(imgs[train_index]), torch.from_numpy(imgs[test_index])
            y_train, y_test = torch.from_numpy(labels[train_index]), torch.from_numpy(labels[test_index])
            mdlPair = train(args, X_train, y_train, X_test, y_test, mdlConst)
            models.append(mdlPair)
            if not args.useENS:
                print('No cross validation')
                break
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()

    if not args.useENS:
        sys.exit(0)
    preds = runEnsemble(mdlConst, models, imgs)
    avg = np.mean(preds.numpy() == labels)
    print('Ensemble average on training set: ', avg)
    preds = runEnsemble(mdlConst, models, img_test)
    df = pd.DataFrame(preds.numpy())
    df.columns = ['Class']
    df.Class = df.Class.astype(int)
    df.index.name = 'Id'
    df.to_csv(strftime("%Y-%m-%d_%H:%M:%S", gmtime())+'_test.csv')

