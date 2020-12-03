import argparse
# import deepspeed

def get_args():
    parser = argparse.ArgumentParser(description='Split_and_Bridge')
    parser.add_argument('--dataset', default='', type=str, required=True,
                        choices=['CIFAR100','TinyImagenet'],help='(default=%(default)s)')
    parser.add_argument('--trainer', default='', type=str, required=True,
                        choices=['split','icarl', 'bic', 'wa', 'dd'],help='(default=%(default)s)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers in Dataloaders')
    parser.add_argument('--nepochs', type=int, default=200, help='Number of epochs for each increment')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1. Note that lr is decayed by args.gamma parameter args.schedule ')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--base-classes', type=int, default=20, help='Number of base classes')
    parser.add_argument('--step-size', type=int, default=20, help='How many classes to add in each increment')
    parser.add_argument('--memory-budget', type=int, default=2000,
                        help='How many images can we store at max. 0 will result in fine-tuning')
    parser.add_argument('--rho', type=float, default=1, help='adaptive split hyperparameter')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds values to be used; seed introduces randomness by changing order of classes')


    args=parser.parse_args()

    return args