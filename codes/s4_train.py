# coding: utf-8
import os
import time
import numpy as np
from tqdm import tqdm
import argparse
import setproctitle
import random
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

import myTransforms
import myModelVgg
from myDataReader import SCdataset
from myUtils import NetPrediction, EvalMetrics


def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-E', '--epoches', default=200, type=int, required=False, help='Epoch, default is 200.')
    parser.add_argument('-B', '--batch_size', default=1000, type=int, required=False, help='batch size, default is 1000.')
    parser.add_argument('-LR', '--initLR', default=0.005, type=float, required=False, help='init lr, default is 0.005.')
    parser.add_argument('-Wg', '--weights', default=None, type=list, required=False, help='weights for CEloss.')
    #weights for loss; or weights = None

    parser.add_argument('-trainpath', '--trainpath', default='../data/train.txt', type=str,
                        required=False, help='trainpath, default is train.txt.')
    parser.add_argument('-validpath', '--validpath', default='../data/val.txt', type=str,
                        required=False, help='valpath, default is val.txt.')
    parser.add_argument('-restore', '--restore', default='', type=str, required=False,
                        help='Model path restoring for testing, if none, just \'\', no default.')
    parser.add_argument('-sn', '--savename', default='../models/FastWSI_vgg_epoch_', type=str, required=False,
                        help='savename for model saving, default is ../models/.')

    parser.add_argument('-net', '--net', default='myVGG', type=str,
                        required=False, help='network from torchvision for classification, default is myVGG')
    parser.add_argument('-loss', '--loss', default='CrossEntropyLoss', type=str,
                        required=False, help='loss function for classification, default is CrossEntropyLoss')

    parser.add_argument('-W', '--nWorker', default=8, type=int, required=False, help='Num worker for dataloader, default is 8.')
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, required=False, help='momentum, default is 0.9.')
    parser.add_argument('-de', '--decay', default=1e-5, type=float, required=False, help='decay, default is 1e-5.')
    parser.add_argument('-C', '--nCls', default=2, type=int, required=False, help='num of Class, here is 2.')
    parser.add_argument('-S', '--seed', default=3, type=int, required=False, help='random seed, default 3.')
    parser.add_argument('-G', '--gpu', default='0', type=str, required=False, help='one or multi gpus, default is 0.')

    args = parser.parse_args()
    with open('../results/FastWSI_vgg_setting.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    #with open('setting.txt', 'r') as f:
    #    args.__dict__ = json.load(f)
    return args


def AdjustLR(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES,power), 8)


def main(args):
    print(args.__dict__)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    normMean = [0.6270, 0.5013, 0.7519]
    normStd = [0.1627, 0.1682, 0.0977]
    ####################### transformer defination, dataset reader and loader
    preprocess = myTransforms.Compose([
        myTransforms.Resize((50, 50)),
        myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                                   myTransforms.RandomVerticalFlip(p=1),
                                   myTransforms.AutoRandomRotation()]),  # above is for: randomly selecting one for process
        # myTransforms.RandomAffine(degrees=[-180, 180], translate=[0., 1.], scale=[0., 2.], shear=[-180, 180, -180, 180]),
        myTransforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0.8, 1.2), hue=0.2),
                                   myTransforms.HEDJitter(theta=0.02)]),
        myTransforms.RandomElastic(alpha=2, sigma=0.06),
        myTransforms.ToTensor(),  #operated on original image, rewrite on previous transform.
        myTransforms.Normalize(normMean, normStd)
    ])
    valpreprocess = myTransforms.Compose([myTransforms.Resize((50,50)),
                                       myTransforms.ToTensor(),
                                       myTransforms.Normalize(normMean,normStd)])

    print('####################Loading dataset...')
    trainset = SCdataset(args.trainpath, preprocess)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.nWorker)
    valset = SCdataset(args.validpath, valpreprocess)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.nWorker)

    net = getattr(myModelVgg, args.net)(in_channels=3, num_classes=args.nCls)
    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    if args.restore:
        net.load_state_dict(torch.load(args.restore)) # load the finetune weight parameters
        print('####################Loading model...', args.restore)
    # else:
    #     net_state_dict = net.state_dict() # get the new network dict
        #pretrained_dict = torch.load('/home/cyyan/.cache/torch/checkpoints/resnet34-333f7ec4.pth') # load the pretrained model
        # pretrained_dict = torch.load('/home/cyyan/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth') # load the pretrained model
        # pretrained_dict_new = {k: v for k, v in pretrained_dict.items() if k in net_state_dict and net_state_dict[k].size() == v.size()} #check the same key in dict.items
        # net_state_dict.update(pretrained_dict_new) # update the new network dict by new dict in pretrained
        # net.load_state_dict(net_state_dict) # load the finetune weight parameters
        # print('####################Loading pretrained model from torch cache checkpoints...')

    print('####################Loading criterion and optimizer...')
    weights = args.weights if args.weights is None else torch.tensor(args.weights).cuda()
    criterion = getattr(nn, args.loss)(weight=weights).cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.initLR, momentum=args.momentum, weight_decay=args.decay)

    print('####################Start training...')
    for epoch in range(args.epoches):
        start = time.time()
        net.train()
    
        AdjustLR(optimizer, epoch, args.epoches, args.initLR, power=0.9)
        print('Current LR:', optimizer.param_groups[0]['lr'])

        losses = 0.0
        for i, (img, label, _) in enumerate(trainloader):
            img = img.cuda()
            label = label.cuda().long()

            output = net(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            print('Iteration {:3d} loss {:.6f}'.format(i + 1, loss.item()))
            setproctitle.setproctitle("Iteration:{}/{}".format(i+1,int(trainset.__len__()/args.batch_size)))
        print('Epoch{:3d}--Time(s){:.2f}--Avgloss{:.4f}-'.format(epoch, time.time()-start, losses/(i+1)))

        torch.save(net.state_dict(), args.savename + str(epoch) + '.pkl')
        print('Model has been saved!')

        real, _, prediction, _ = NetPrediction(valloader, net)
        result = EvalMetrics(real, prediction)
        for key in result:
            print(key, ': ', result[key])
    print('####################Finished Training!')


if __name__ == '__main__':
    arg = GetArgs()
    main(arg)
