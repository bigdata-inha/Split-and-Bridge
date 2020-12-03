import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, num_class):
        
        if dataset == 'CIFAR100':
            
            import networks.MyNetwork_split as res
            return res.network('CIFAR',32,num_class)

        if dataset == 'TinyImagenet':

            import networks.MyNetwork_split as res
            return res.network('TinyImagenet', 64, num_class)

