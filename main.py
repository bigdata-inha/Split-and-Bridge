import argparse
import torch
import torch.utils.data as td
import numpy as np
import random
import data_handler
import networks
import trainer
import arguments
import utils.utils
# import deepspeed
from sklearn.utils import shuffle

torch.multiprocessing.set_start_method('spawn',force=True)

if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = arguments.get_args()
    log_name = '{}_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
        args.trainer,
        args.seed,
        args.base_classes,
        args.step_size,
        args.batch_size,
        args.nepochs,
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = args.seed
    m = args.memory_budget

    # Fix the seed.
    args.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
    if args.dataset == 'CIFAR100':
        loader = None
    else:
        loader = dataset.loader

    # Loader used for training data ####################################################
    shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)
    print(shuffle_idx)
    train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                          dataset.train_labels,
                                                          dataset.classes,
                                                          args.step_size,
                                                          args.memory_budget,
                                                          'train',
                                                          transform=dataset.train_transform,
                                                          loader=loader,
                                                          shuffle_idx=shuffle_idx,
                                                          base_classes=args.base_classes,
                                                          approach=args.trainer
                                                          )
    # Loader for evaluation
    evaluate_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                             dataset.train_labels,
                                                             dataset.classes,
                                                             args.step_size,
                                                             args.memory_budget,
                                                             'train',
                                                             transform=dataset.train_transform,
                                                             loader=loader,
                                                             shuffle_idx=shuffle_idx,
                                                             base_classes=args.base_classes,
                                                             approach='ft'
                                                             )

    # Loader for test data.
    test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data,
                                                         dataset.test_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         args.memory_budget,
                                                         'test',
                                                         transform=dataset.test_transform,
                                                         loader=loader,
                                                         shuffle_idx=shuffle_idx,
                                                         base_classes=args.base_classes,
                                                         approach=args.trainer
                                                         )

    result_dataset_loaders = data_handler.make_ResultLoaders(dataset.test_data,
                                                             dataset.test_labels,
                                                             dataset.classes,
                                                             args.step_size,
                                                             transform=dataset.test_transform,
                                                             loader=loader,
                                                             shuffle_idx=shuffle_idx,
                                                             base_classes=args.base_classes
                                                             )

    # Iterator to iterate over training data.##################################################
    train_iterator = data_handler.iterator(train_dataset_loader, batch_size=args.batch_size, shuffle=True, drop_last=True)

    evaluator_iterator = data_handler.iterator(evaluate_dataset_loader, batch_size=args.batch_size, shuffle=True)

    # Iterator to iterate over test data
    test_iterator = data_handler.iterator(test_dataset_loader, batch_size=100, shuffle=False)

    # Get the required model########################################################################
    myModel = networks.ModelFactory.get_model(args.dataset, args.base_classes)

    # Define the optimizer used in the experiment
    optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)
    # Trainer object used for training
    myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)

    # Initilize the evaluators used to measure the performance of the system.
    testType = 'trainedClassifier'
    t_classifier = trainer.EvaluatorFactory.get_evaluator(testType, classes=dataset.classes)

    # Loop that incrementally adds more and more classes #####################################

    print(args.step_size)

    train_start = 0
    train_end = args.base_classes
    test_start = 0
    test_end = args.base_classes
    total_epochs = args.nepochs
    schedule = np.array(args.schedule)
    balance_factor = 0

    tasknum = (dataset.classes - args.base_classes) // args.step_size + 1

    results = {}
    for head in ['all']:
        results[head] = {}
        results[head]['correct'] = []
        results[head]['stat'] = []

    results['task_soft_1'] = np.zeros((tasknum, tasknum))

    print(tasknum)

    # iterate for each task ####################################################################
    for t in range(tasknum):

        results['all']['correct'] = []
        results['all']['stat'] = []

        if t > 0:
            optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay,
                                        nesterov=True)
            myTrainer.get_optimizer(optimizer)
            myTrainer.update_frozen_model()

        print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)

        # Add new classes to the train, and test iterator
        lr = args.lr
        schedule = args.schedule
        myTrainer.setup_training(lr)

        flag = 0
        if args.trainer == 'split' and t == 0:
            try:
                model_name = 'models/trained_model/split_{}_base_{}_step_{}_batch_{}_epoch_{}_task_{}.pt'.format(
                     args.seed, args.base_classes, args.step_size, args.batch_size, args.nepochs, t)
                myTrainer.model.load_state_dict(torch.load(model_name))
                flag = 1
            except:
                pass

        best_acc = 0
        best_model = None

        # Running nepochs epochs   step 1 ###############################################################################
        print('Flag: %d' % flag)
        for epoch in range(0, total_epochs):
            if flag or balance_factor == 1:
                break
            myTrainer.update_lr(epoch, schedule)
            myTrainer.first_train(epoch)

            train_1 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)

            if epoch % 5 == (4):
                if t == 0:
                    test_1 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                           mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax): %0.2f" % test_1)

                    if test_1 > best_acc:
                        best_acc = test_1
                        best_model = utils.get_model(myModel)
                        print("change best model")

                else:
                    correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                          test_start, test_end,
                                                          mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
                    print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
                    print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
                    print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
                    print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

                    if correct['intra_pre'] > best_acc:
                        best_acc = correct['intra_pre']
                        best_model = utils.get_model(myModel)
                        print("change best model")


        if flag == 0 and balance_factor != 1:
            utils.set_model_(myModel, best_model)

        if t > 0 and balance_factor != 1:
            results_1 = {}
            for head in ['all']:
                results_1[head] = {}
                results_1[head]['correct'] = []
                results_1[head]['stat'] = []

            correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                  test_start, test_end,
                                                  mode='test', step_size=args.step_size)
            for head in ['all', 'pre', 'new', 'intra_pre', 'intra_new']:
                results_1['all']['correct'].append(correct[head])
            results_1['all']['stat'].append(stat['all'])

            print(results_1)

        # split ###################################################################################
        if t > 0 and balance_factor != 1:
            myTrainer.split()
            print(myTrainer.model)

        # Running nepochs epochs   step 2 #########################################################
        if t > 0 and balance_factor != 1:
            best_acc = 0
            best_model = None

            lr = args.lr
            schedule = args.schedule
            myTrainer.setup_training(lr)

            optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay
                                    , nesterov=True)
            myTrainer.get_optimizer(optimizer)

        for epoch in range(0, total_epochs):
            if t == 0 or balance_factor == 1:
                break

            myTrainer.update_lr(epoch, schedule)
            myTrainer.second_train(epoch)

            train_1 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)

            if epoch % 5 == (4):

                if t == 0:
                    test_1 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                                   mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax): %0.2f" % test_1)

                else:
                    correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                          test_start, test_end,
                                                          mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
                    print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
                    print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
                    print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
                    print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

                    if correct['intra_pre'] > best_acc:
                        best_acc = correct['intra_pre']
                        best_model = utils.get_model(myModel)
                        print("change best model")

        if flag == 0 and t > 0:
            if balance_factor != 1:
                utils.set_model_(myModel, best_model)

        if t > 0 and balance_factor != 1:
            results_2 = {}
            for head in ['all']:
                results_2[head] = {}
                results_2[head]['correct'] = []
                results_2[head]['stat'] = []

            correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                  test_start, test_end,
                                                  mode='test', step_size=args.step_size)
            for head in ['all', 'pre', 'new', 'intra_pre', 'intra_new']:
                results_2['all']['correct'].append(correct[head])
            results_2['all']['stat'].append(stat['all'])

            print(results_2)

        # update frozen_model ##############################################################
        if t > 0 and balance_factor != 1:
            myTrainer.update_frozen_model()

        # reunion ############################################################################
        if t > 0 and balance_factor != 1:
            myTrainer.reunion()

        # Running nepochs epochs   step 3 ####################################################
        best_acc = 0
        best_model = None

        if t > 0:
            lr = args.lr
            schedule = args.schedule
            myTrainer.setup_training(lr)

            optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.decay, nesterov=True)
            myTrainer.get_optimizer(optimizer)

        for epoch in range(0, total_epochs):
            if t == 0:
                break
            myTrainer.update_lr(epoch, schedule)
            myTrainer.third_train(epoch)

            train_1 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d" % epoch)
            print("Train Classifier top-1 (Softmax): %0.2f" % train_1)

            if epoch % 5 == (4):

                if t == 0:
                    test_1 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                                       mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax): %0.2f" % test_1)
                    if test_1 > best_acc:
                        best_acc = test_1
                        best_model = utils.get_model(myModel)
                        print("change best model")

                else:
                    correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                              test_start, test_end,
                                                              mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
                    print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
                    print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
                    print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
                    print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

                    if correct['all'] > best_acc:
                        best_acc = correct['all']
                        best_model = utils.get_model(myModel)
                        print("change best model")

        if t > 0:
            utils.set_model_(myModel, best_model)

            results_1 = {}
            for head in ['all']:
                results_1[head] = {}
                results_1[head]['correct'] = []
                results_1[head]['stat'] = []

            correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                  test_start, test_end,
                                                  mode='test', step_size=args.step_size)
            for head in ['all', 'pre', 'new', 'intra_pre', 'intra_new']:
                results_1['all']['correct'].append(correct[head])
            results_1['all']['stat'].append(stat['all'])

            print(results_1)

        # weight align ########################################################################
        if t > 0:
            myTrainer.weight_align()
            correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                  test_start, test_end,
                                                  mode='test', step_size=args.step_size)
            print("Test Classifier top-1 (Softmax, all): %0.2f" % correct['all'])
            print("Test Classifier top-1 (Softmax, pre): %0.2f" % correct['pre'])
            print("Test Classifier top-1 (Softmax, new): %0.2f" % correct['new'])
            print("Test Classifier top-1 (Softmax, intra_pre): %0.2f" % correct['intra_pre'])
            print("Test Classifier top-1 (Softmax, intra_new): %0.2f" % correct['intra_new'])

        if t > 0:
            correct, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                  test_start, test_end,
                                                  mode='test', step_size=args.step_size)
            for head in ['all', 'pre', 'new', 'intra_pre', 'intra_new']:
                results['all']['correct'].append(correct[head])
            results['all']['stat'].append(stat['all'])

        else:
            test_1 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end,
                                           mode='test', step_size=args.step_size)
            print("Test Classifier top-1 (Softmax): %0.2f" % test_1)
            for head in ['all']:
                results[head]['correct'].append(test_1)

        start = 0
        end = args.base_classes
        for i in range(t + 1):
            dataset_loader = result_dataset_loaders[i]
            iterator = data_handler.iterator(dataset_loader, batch_size=args.batch_size)

            results['task_soft_1'][t][i] = t_classifier.evaluate(myTrainer.model,iterator, start, end)
            start = end
            end += args.step_size

        print(results)

        f = open('./result_data/{}_task_{}_output.txt'.format(log_name, t), "w")
        f.write(str(results))
        f.close()

        torch.save(myModel.state_dict(), "C:/Users/admin/Desktop/Split_and_Bridge/models/trained_ model/{}_task_{}.pt".format(log_name, t))

        # incremental task ############################################################################
        if t != tasknum - 1:
            myTrainer.increment_classes()
            evaluate_dataset_loader.update_exemplar()
            evaluate_dataset_loader.task_change()

            train_end = train_end + args.step_size
            test_end = test_end + args.step_size

            ratio = ((train_end - args.step_size)/train_end)
            balance_factor = min(1,ratio * args.rho)

            myModel.Incremental_learning(test_end, args.step_size, balance_factor)


