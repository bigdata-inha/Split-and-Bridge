import logging

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def make_pred(out, start, end, step_size):
    pred = {}
    pred_5 = {}
    pred['all'] = out.data.max(1, keepdim=True)[1]
    pred_5['all'] = torch.topk(out, 5, dim=1)[1]

    prev_out = out[:,start:end-step_size]
    curr_out = out[:,end-step_size:end]

    prev_soft = F.softmax(prev_out, dim=1)
    curr_soft = F.softmax(curr_out, dim=1)

    output = torch.cat((prev_soft, curr_soft), dim=1)

    pred['prev_new'] = output.data.max(1, keepdim=True)[1]
    pred_5['prev_new'] = torch.topk(output, 5, dim=1)[1]

    soft_arr = []
    for t in range(start,end,step_size):
        temp_out = out[:,t:t+step_size]
        temp_soft = F.softmax(temp_out, dim=1)
        soft_arr += [temp_soft]

    output = torch.cat(soft_arr, dim=1)

    pred['task'] = output.data.max(1, keepdim=True)[1]
    pred_5['task'] = torch.topk(output, 5, dim=1)[1]
    
    return pred, pred_5

def cnt_stat(target, start, end, step_size, mode, head, pred, pred_5, correct, correct_5, stat, batch_size):
    
    correct[head] += pred[head].eq(target.data.view_as(pred[head])).sum().item()
    correct_5[head] += pred_5[head].eq(target.data.unsqueeze(1).expand(pred_5[head].shape)).sum().item()

    if mode == 'prev':
        cp_ = pred[head].eq(target.data.view_as(pred[head])).sum()
        epn_ = (pred[head] >= end-step_size).int().sum()
        epp_ = (batch_size-(cp_ + epn_))
        stat[head][0] += cp_.item()
        stat[head][1] += epp_.item()
        stat[head][2] += epn_.item()
    else:
        cn_ = pred[head].eq(target.data.view_as(pred[head])).sum()
        enp_ = (pred[head] < end-step_size).int().sum()
        enn_ = (batch_size-(cn_ + enp_))
        stat[head][3] += cn_.item()
        stat[head][4] += enn_.item()
        stat[head][5] += enp_.item()
    return

def cheat(out, target, start, end, mod, correct, correct_5):
    output = out[:,start:end]
    target = target % (mod)
    
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    ans = pred.eq(target.data.view_as(pred)).sum()
    correct['cheat'] += ans.item()

    pred_5 = torch.topk(output, 5, dim=1)[1]
    ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum()
    correct_5['cheat'] += ans.item()

class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass
    

    @staticmethod
    def get_evaluator(testType="trainedClassifier", classes=1000, option='euclidean'):
        if testType == "trainedClassifier":
            return softmax_evaluator()
        if testType == "bic":
            return BiC_evaluator(classes)
        if testType == "generativeClassifier":
            return GDA(classes, option = option)

class GDA():

    def __init__(self, classes, option = 'euclidean'):
        self.classes = classes
        self.option = option
    
    def update_moment(self, model, loader, step_size, tasknum):
        
        model.eval()
        with torch.no_grad():
            # compute means
            classes = step_size * (tasknum+1)
            class_means = torch.zeros((classes,512)).cuda()
            totalFeatures = torch.zeros((classes, 1)).cuda()
            total = 0
            # Iterate over all train Dataset
            for data, target in loader:
                data, target = data.cuda(), target.long().cuda()
                #self.batch_size = data.shape[0]
                if data.shape[0]<4:
                    continue
                total += data.shape[0]
                try:
                    _, features = model.forward(data, feature_return=True)
                except:
                    continue
                    
                class_means.index_add_(0, target, features.data)
                totalFeatures.index_add_(0, target, torch.ones_like(target.unsqueeze(1)).float().cuda())
                
            class_means = class_means / totalFeatures
            
            # compute precision
            covariance = torch.zeros(512,512).cuda()
            euclidean = torch.eye(512).cuda()

            if self.option == 'Mahalanobis':
                for data, target in tqdm(loader):
                    data, target = data.cuda(), target.long().cuda()
                    _, features = model.forward(data, feature_return=True)

                    vec = (features.data - class_means[target])
                    
                    np.expand_dims(vec, axis=2)
                    cov = torch.matmul(vec.unsqueeze(2), vec.unsqueeze(1)).sum(dim=0)
                    covariance += cov

                #avoid singular matrix
                covariance = covariance / totalFeatures.sum() + torch.eye(512).cuda() * 1e-9
                precision = covariance.inverse()

            self.class_means = class_means
            if self.option == 'Mahalanobis':
                self.precision = precision
            else:
                self.precision = euclidean
        
            return
    
    def evaluate(self, model, loader, start, end, mode='train', step_size=100):
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            self.start = start
            self.end = end
            self.step_size = step_size
            self.stat = {}
            self.correct = {}
            self.correct['intra_pre'] = 0
            self.correct['intra_new'] = 0
            head_arr = ['all', 'pre', 'new']

            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                self.stat[head] = [0,0,0,0,0,0,0]
                self.correct[head] = 0

            for data, target in loader:
                data, target = data.cuda(), target.long().cuda()
                if data.shape[0]<4:
                    continue
                try:
                    _, features = model.forward(data, feature_return=True)
                except:
                    continue

                self.batch_size = data.shape[0]
                total += data.shape[0]

                batch_vec = (features.data.unsqueeze(1) - self.class_means.unsqueeze(0))
                temp = torch.matmul(batch_vec, self.precision)
                out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).squeeze()
                


                if mode == 'test' and end > step_size:

                    self.make_pred(out)

                    if target[0]<end-step_size: # prev

                        self.cnt_stat(target, 'prev', 'all')

                        output = out[:, start:end - step_size]
                        target = target % (end - start - step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['intra_pre'] += ans

                    else: # new

                        self.cnt_stat(target, 'new', 'all')

                        output = out[:, end - step_size:end]
                        target = target % (step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['intra_new'] += ans

                else:
                    target = target % (end - start)
                    pred = out.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).sum().item()

            if mode == 'test' and end > step_size:

                for head in ['all']:
                    self.correct[head] = 100. * self.correct[head] / total
                self.stat['all'][6] = total
                self.correct['pre'] = 100 * self.stat['all'][0] / (
                            self.stat['all'][0] + self.stat['all'][1] + self.stat['all'][2])
                self.correct['new'] = 100 * self.stat['all'][3] / (
                            self.stat['all'][3] + self.stat['all'][4] + self.stat['all'][5])
                self.correct['intra_pre'] = 100 * self.correct['intra_pre'] / (
                            self.stat['all'][0] + self.stat['all'][1] + self.stat['all'][2])
                self.correct['intra_new'] = 100 * self.correct['intra_new'] / (
                            self.stat['all'][3] + self.stat['all'][4] + self.stat['all'][5])
                return self.correct, self.stat

            return 100. * correct / total

    def make_pred(self, out):
        start, end, step_size = self.start, self.end, self.step_size
        self.pred = {}
        self.pred['all'] = out.data.max(1, keepdim=True)[1]
        return

    def cnt_stat(self, target, mode, head):
        start, end, step_size = self.start, self.end, self.step_size
        pred = self.pred[head]
        self.correct[head] += pred.eq(target.data.view_as(pred)).sum().item()

        if mode == 'prev':
            cp_ = pred.eq(target.data.view_as(pred)).sum()
            epn_ = (pred >= end - step_size).int().sum()
            epp_ = (self.batch_size - (cp_ + epn_))
            self.stat[head][0] += cp_.item()
            self.stat[head][1] += epp_.item()
            self.stat[head][2] += epn_.item()
        else:
            cn_ = pred.eq(target.data.view_as(pred)).cpu().sum()
            enp_ = (pred.cpu().numpy() < end - step_size).sum()
            enn_ = (self.batch_size - (cn_ + enp_))
            self.stat[head][3] += cn_.item()
            self.stat[head][4] += enn_.item()
            self.stat[head][5] += enp_.item()
        return
        

class BiC_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self, classes):
        
        self.classes = classes
        
    def evaluate(self, model, loader, start, end, bias_correction_layer, mode='train', step_size=100):
        
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            self.start = start
            self.end = end
            self.step_size = step_size
            self.stat = {}
            self.correct = {}
            self.correct['intra_pre'] = 0
            self.correct['intra_new'] = 0
            head_arr = ['all', 'pre', 'new']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                self.stat[head] = [0,0,0,0,0,0,0]
                self.correct[head] = 0

            for data, target in loader:
                data, target = data.cuda(), target.long().cuda()

                self.batch_size = data.shape[0]
                total += data.shape[0]

                out = model(data)[:,:end]
                if end > step_size:
                    out_new = bias_correction_layer(out[:,end-step_size:end])
                    out = torch.cat((out[:,:end-step_size], out_new), dim=1)

                if mode == 'test' and end > step_size:
                    
                    pred = self.make_pred(out)

                    if target[0]<end-step_size: # prev

                        self.cnt_stat(target, 'prev', 'all')

                        output = out[:, start:end - step_size]
                        target = target % (end - start - step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['intra_pre'] += ans

                    else: # new

                        self.cnt_stat(target, 'new', 'all')

                        output = out[:, end - step_size:end]
                        target = target % (step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['intra_new'] += ans
                else:
                    output = model(data)[:,start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).sum().item()



            if mode == 'test' and end > step_size:

                for head in ['all']:
                    self.correct[head] = 100. * self.correct[head] / total
                self.stat['all'][6] = total
                self.correct['pre'] = 100 * self.stat['all'][0] / (
                            self.stat['all'][0] + self.stat['all'][1] + self.stat['all'][2])
                self.correct['new'] = 100 * self.stat['all'][3] / (
                            self.stat['all'][3] + self.stat['all'][4] + self.stat['all'][5])
                self.correct['intra_pre'] = 100 * self.correct['intra_pre'] / (
                            self.stat['all'][0] + self.stat['all'][1] + self.stat['all'][2])
                self.correct['intra_new'] = 100 * self.correct['intra_new'] / (
                            self.stat['all'][3] + self.stat['all'][4] + self.stat['all'][5])
                return self.correct, self.stat

            return 100. * correct / total

    def make_pred(self, out):
        start, end, step_size = self.start, self.end, self.step_size
        self.pred = {}
        self.pred['all'] = out.data.max(1, keepdim=True)[1]
        return

    def cnt_stat(self, target, mode, head):
        start, end, step_size = self.start, self.end, self.step_size
        pred = self.pred[head]
        self.correct[head] += pred.eq(target.data.view_as(pred)).sum().item()

        if mode == 'prev':
            cp_ = pred.eq(target.data.view_as(pred)).sum()
            epn_ = (pred >= end - step_size).int().sum()
            epp_ = (self.batch_size - (cp_ + epn_))
            self.stat[head][0] += cp_.item()
            self.stat[head][1] += epp_.item()
            self.stat[head][2] += epn_.item()
        else:
            cn_ = pred.eq(target.data.view_as(pred)).cpu().sum()
            enp_ = (pred.cpu().numpy() < end - step_size).sum()
            enn_ = (self.batch_size - (cn_ + enp_))
            self.stat[head][3] += cn_.item()
            self.stat[head][4] += enn_.item()
            self.stat[head][5] += enp_.item()
        return
    
class softmax_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self):
        pass

    def evaluate(self, model, loader, start, end, mode='train', step_size=100):
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            self.start = start
            self.end = end
            self.step_size = step_size
            self.stat = {}
            self.correct = {}
            self.correct['intra_pre'] = 0
            self.correct['intra_new'] = 0
            head_arr = ['all', 'pre', 'new']
            for head in head_arr:
                # cp, epp, epn, cn, enn, enp, total
                self.stat[head] = [0,0,0,0,0,0,0]
                self.correct[head] = 0


            for data, target in loader:
                data, target = data.cuda(), target.long().cuda()

                self.batch_size = data.shape[0]
                total += data.shape[0]

                if mode == 'test' and end > step_size:

                    out = model(data)
                    self.make_pred(out)

                    if target[0]<end-step_size: # prev

                        self.cnt_stat(target, 'prev', 'all')

                        output = out[:,start:end-step_size]
                        target = target % (end - start-step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['intra_pre'] += ans

                    else: # new

                        self.cnt_stat(target, 'new', 'all')

                        output = out[:,end-step_size:end]
                        target = target % (step_size)

                        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                        ans = pred.eq(target.data.view_as(pred)).sum().item()
                        self.correct['intra_new'] += ans


                else:
                    output = model(data)[:,start:end]
                    target = target % (end - start)

                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).sum().item()


            if mode == 'test' and end > step_size:

                for head in ['all']:
                    self.correct[head] = 100. * self.correct[head] / total
                self.stat['all'][6] = total
                self.correct['pre'] = 100 * self.stat['all'][0] / (self.stat['all'][0] + self.stat['all'][1] + self.stat['all'][2])
                self.correct['new'] = 100 * self.stat['all'][3] / (self.stat['all'][3] + self.stat['all'][4] + self.stat['all'][5])
                self.correct['intra_pre'] = 100 * self.correct['intra_pre'] / (self.stat['all'][0] + self.stat['all'][1] + self.stat['all'][2])
                self.correct['intra_new'] = 100 * self.correct['intra_new'] / (self.stat['all'][3] + self.stat['all'][4] + self.stat['all'][5])
                return self.correct, self.stat

            return     100. * correct / total

    def make_pred(self, out):
        start, end, step_size = self.start, self.end, self.step_size
        self.pred = {}
        self.pred['all'] = out.data.max(1, keepdim=True)[1]
        return

    def cnt_stat(self, target, mode, head):
        start, end, step_size = self.start, self.end, self.step_size
        pred = self.pred[head]
        self.correct[head] += pred.eq(target.data.view_as(pred)).sum().item()

        if mode == 'prev':
            cp_ = pred.eq(target.data.view_as(pred)).sum()
            epn_ = (pred >= end - step_size).int().sum()
            epp_ = (self.batch_size - (cp_ + epn_))
            self.stat[head][0] += cp_.item()
            self.stat[head][1] += epp_.item()
            self.stat[head][2] += epn_.item()
        else:
            cn_ = pred.eq(target.data.view_as(pred)).cpu().sum()
            enp_ = (pred.cpu().numpy() < end - step_size).sum()
            enn_ = (self.batch_size - (cn_ + enp_))
            self.stat[head][3] += cn_.item()
            self.stat[head][4] += enn_.item()
            self.stat[head][5] += enp_.item()
        return