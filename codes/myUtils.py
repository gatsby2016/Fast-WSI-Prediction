import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def NetPrediction(dataloader, model):
    real = np.array([])
    score = np.array([])
    predictions = np.array([])
    namelist = np.array([])

    model.eval()
    with torch.no_grad():
        for i, (img, target, name) in tqdm(enumerate(dataloader)):
            out = model(img.cuda())

            prob = F.softmax(out, 1, _stacklevel=5)
            pred = torch.argmax(prob, dim=1)

            real = np.concatenate((real, target), axis=0)
            predictions = np.concatenate((predictions, pred.cpu().numpy()), axis=0)
            score = np.concatenate((score, prob.cpu().numpy()[:, 1]), axis=0)
            namelist = np.concatenate((namelist, name), axis=0)
    return real, score, predictions, namelist


def EvalMetrics(real, prediction):
    TP = ((real == 1) & (prediction == 1)).sum()  # label 1 is positive
    FN = ((real == 1) & (prediction == 0)).sum()
    TN = ((real == 0) & (prediction == 0)).sum()  # label 0 is negtive
    FP = ((real == 0) & (prediction == 1)).sum()
    print('==============================')
    print('          |  predict          ')
    print('          |  Postive  Negtive ')
    print('==============================')
    print('  Postive | ', TP, '  ', FN, '     = ', TP + FN)
    print('  Negtive | ', FP, '  ', TN, '     = ', TN + FP)
    print('==============================')
    res = {}
    res['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
    res['Specificity'] = TN / (TN + FP)
    res['Recall'] = TP / (TP + FN)
    res['Precision'] = TP / (TP + FP)
    res['F1Score'] = (2 * res['Recall'] * res['Precision']) / (res['Recall'] + res['Precision'])
    #    return [Accuracy, Specificity, Recall, Precision, F1Score]
    return res