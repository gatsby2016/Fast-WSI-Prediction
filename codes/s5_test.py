# coding: utf-8
import os
import numpy as np
import random
import argparse

import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

import myTransforms
import myModelVgg
from myDataReader import SCdataset
from myUtils import NetPrediction, EvalMetrics


def GetArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', '--batch_size', default=2000, type=int, required=False, help='batch size, default is 2000.')
    parser.add_argument('-testpath', '--testpath', default='../data/test.txt', type=str,
                        required=False, help='valpath, default is test.txt.')
    parser.add_argument('-restore', '--restore', default='../models/FastWSI_vgg_epoch_189.pkl', type=str, required=False,
                        help='Model path restoring for testing, if none, just \'\', no default.')
    parser.add_argument('-sn', '--savename', default='../results/test_pred_score.npz', type=str, required=False,
                        help='savename for model saving, default is ../results/test_pred_score.npz.')

    parser.add_argument('-net', '--net', default='myVGG', type=str,
                        required=False, help='network for classification, default is myVGG')

    parser.add_argument('-W', '--nWorker', default=8, type=int, required=False, help='Num worker for dataloader, default is 8.')
    parser.add_argument('-C', '--nCls', default=2, type=int, required=False, help='num of Class, here is 2.')
    parser.add_argument('-S', '--seed', default=3, type=int, required=False, help='random seed, default 3.')
    parser.add_argument('-G', '--gpu', default='0', type=str, required=False, help='one or multi gpus, default is 0.')

    args = parser.parse_args()
    return args


def main(args):
    print(args.__dict__)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    normMean = [0.6270, 0.5013, 0.7519]
    normStd = [0.1627, 0.1682, 0.0977]
    preprocess = myTransforms.Compose([myTransforms.Resize((50, 50)),
                                       myTransforms.ToTensor(),  #operated on original image
                                       myTransforms.Normalize(normMean, normStd)])

    testset = SCdataset(args.testpath, preprocess)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.nWorker)

    net = getattr(myModelVgg, args.net)(in_channels=3, num_classes=args.nCls)
    if len(args.gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    if args.restore:
        net.load_state_dict(torch.load(args.restore)) # load the finetune weight parameters

    real, score, prediction, namelist = NetPrediction(testloader, net)
    result = EvalMetrics(real, prediction)
    for key in result:
        print(key, ': ', result[key])

    np.savez(args.savename, key_real=real, key_score=score,key_pred=prediction, key_namelist=namelist)


if __name__ == '__main__':
    arg = GetArgs()
    main(arg)
