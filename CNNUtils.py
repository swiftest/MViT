import torch
import itertools
import numpy as np
import torch.nn as nn
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    # output: (batch, 13)
    # target: (batch, )
    maxk = max(topk)  # 1
    batch_size = target.size(0)  # batch
    
    _, pred = output.topk(maxk, axis=1)  # (batch, 1) or (batch, 2) or (batch, 3)
    pred = pred.t()  # (1, batch) or (2, batch) or (3, batch), it is used to predict the 1st, 2nd, and 3rd most likely categories
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # (1, batch) or (2, batch) or (3, batch) 
    
    res = []  # The program here is very important. This res list is used to record the ratio of the total number of accurate predictions made in the previous few times (1, 2, or 3 times) to the total number of samples in the entire batch (i.e. accuracy)!!! Be sure to analyze it carefully!!!
    
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum()
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()


def train(model, train_loader, criterion, optimizer):
    model.train()
    objs = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data)  # (batch, 16)

        loss = criterion(batch_pred, batch_target)

        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)  # 计算所有训练样本的平均损失
        top1.update(prec1[0].data, n)  # 计算所有训练样本的平均准确率
    return top1.avg, objs.avg


def test(model, test_loader):
    model.eval()
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
            
            batch_pred = model(batch_data)  # (batch, 13)
            _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
            pp = pred.squeeze()  # (100, )
            
            tar = np.append(tar, batch_target.data.cpu().numpy())
            pre = np.append(pre, pp.data.cpu().numpy())
    return tar, pre


def valid(model, valid_loader, criterion):
    model.eval()
    objs = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
        
            batch_pred = model(batch_data)  # (100, 13)
            _, pred = batch_pred.topk(1, axis=1)  # (100, 1)
            
            loss = criterion(batch_pred, batch_target)

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.data, n)  # 计算所有训练样本的平均损失
            top1.update(prec1[0].data, n)  # 计算所有训练样本的平均准确率
    return top1.avg, objs.avg
