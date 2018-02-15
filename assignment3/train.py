#!/usr/bin/env python
import torch
import argparse
import dataloader
import numpy as np
import mininet as mn
import torch.utils.data as dl
from torch.autograd import Variable
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('General tool to train a NN based on passed configuration.')
    parser.add_argument('--nSplit', dest='nSplit', default=2, type=int, help='How many splits to use in KFold cross validation.')
    args = parser.parse_args()
    return args
#
# Main loop for running the agent.
def train(args, imgs, labels, img_test, label_test):
    # 
    # Create the image augmentation.
    t = None
    topil = transforms.ToPILImage()
    train = dataloader.npdataset(imgs, labels.view(-1), t)
    stages = {
        'train': torch.utils.data.DataLoader(train, batch_size=16, shuffle=False)
    }
    model = mn.Mininet()
    usegpu = torch.cuda.is_available()
    # 
    # Iterate.
    for stage in stages:
        #
        # Switch on / off gradients.
        model.train(stage == 'train')
        loader = stages[stage]
        for data in loader:
            inputs, labels_cpu = data['img'], data['label']
            if usegpu:
                labels_cpu.squeeze_()
                inputs, labels = Variable(inputs).cuda(async = True), Variable(labels_cpu).cuda(async = True)
            else:
                inputs, labels = Variable(inputs), Variable(labels_cpu)
            print(model(inputs))
            break
#
# Main code.
if __name__ == '__main__':
    args = getInputArgs()
    imgs = np.expand_dims(np.load('data/X.npy').astype(np.float32) / 255, 1)
    # 
    # Image statistics. 
    # curMean = np.mean(imgs, axis=(0,2,3))
    # curVar = np.var(imgs, axis=(0,2,3))
    # print(curMean)
    # print(curVar)
    labels = np.load('data/Y.npy')
    kf = KFold(n_splits=args.nSplit, shuffle = True)
    for train_index, test_index in kf.split(imgs): 
        X_train, X_test = imgs[train_index], imgs[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train(args, torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(X_test), torch.from_numpy(y_test))