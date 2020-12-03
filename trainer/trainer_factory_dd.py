import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td


class TrainerFactory_dd():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer, ref_train_iterator = None, ref_model = None, ref_optimizer = None):

        if args.trainer == 'lwf':
            import trainer.lwf as trainer
        elif args.trainer == 'ssil':
            import trainer.ssil as trainer
        elif args.trainer == 'ft' or args.trainer == 'il2m':
            import trainer.ft as trainer
        elif args.trainer == 'icarl':
            import trainer.icarl as trainer
        elif args.trainer == 'bic':
            import trainer.bic as trainer
        elif args.trainer == 'wa':
            import trainer.wa as trainer
        elif args.trainer == 'dd':
            import trainer.dd as trainer

        return trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer, ref_train_iterator, ref_model, ref_optimizer)


class ExemplarLoader(td.Dataset):
    def __init__(self, train_dataset):

        self.data = train_dataset.data
        self.labels = train_dataset.labels
        self.labelsNormal = train_dataset.labelsNormal
        self.exemplar = train_dataset.exemplar
        self.transform = train_dataset.transform
        self.loader = train_dataset.loader
        self.mem_sz = len(self.exemplar)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        index = self.exemplar[index % self.mem_sz]
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this.
    '''

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        self.train_data_iterator = trainDataIterator
        self.test_data_iterator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.train_loader = self.train_data_iterator.dataset
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.current_lr = args.lr