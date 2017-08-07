# coupled_train.py
# Functions to perform coupled training
# NOTE: Run this file to do coupled training after completion
#       of differential training.


import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_reader import image_data_set
from networks import *
from utils import *
from test_scnn import test_scnn


# Train switch classifier
def train_switch(networks, datasets, train_funcs, test_funcs, lr, log_fp):
    log(log_fp, 'Training switch...')
    
    # Create label set for training switch classifier
    num_classes = len(test_funcs) - 1
    eq_data = [[] for _ in range(num_classes)]
    losses = np.zeros((1, num_classes))

    # get label for every sample
    for X, Y in datasets['train']:
        for i, test_fn in enumerate(test_funcs[1: ]):
            _, Y_pred = test_fn(X, Y)
            losses[0, i] = np.abs(np.sum(Y_pred) - np.sum(Y))
        Y_pc = np.argmin(losses, axis = 1)
        eq_data[Y_pc[0]].append((X, Y_pc))

    # equalise the number of samples across classes
    num_files_per_class = max([len(ds) for ds in eq_data])
    train_data = []
    for i, ds in enumerate(eq_data):
        samples = []
        samples += ds
        while len(samples) < num_files_per_class:
            samples += random.sample(ds,
                min(num_files_per_class - len(samples), len(ds)))
        random.shuffle(samples)                 
        train_data += samples

    # Train switch classifier
    num_epochs = 1
    for epoch in range(num_epochs):
        avg_pc_loss = 0.0
        random.shuffle(train_data)
        for i, (X, Y) in enumerate(train_data):
            pc_loss = train_funcs[0](X, Y, lr)
            avg_pc_loss += pc_loss
            if i % 500 == 0:
                log(log_fp, 'iter: %d, pc_loss: %f, avg_pc_loss: %f' % \
                                (i, pc_loss, avg_pc_loss / (i + 1)))
        avg_pc_loss /= (i + 1)
        log(log_fp, 'done; avg_pc_Loss: %.12f' % (avg_pc_loss))


# Do switched differential training
def train_switched_differential(networks, datasets, train_funcs, test_funcs, 
                                run_funcs, lr, log_fp):
    log(log_fp, 'Switched Differential Training...')
    
    num_epochs = 1
    for epoch in range(num_epochs):
        switch_stat = np.zeros((len(test_funcs) - 1, ))
        for i, (X, Y) in enumerate(datasets['train']):

            # run switch classifier to get the label
            label = run_funcs[0](X)
            Y_pc = np.argmax(label, axis = 1)[0]

            # backpropagate the regressor suggested by the classifier
            loss = train_funcs[Y_pc + 1](X, Y, lr)
            switch_stat[Y_pc] += 1
            if i % 500 == 0:
                txt = ''
                for i in range(switch_stat.shape[0]):
                    txt += 'n%d: %d, ' % (i, switch_stat[i])
                log(log_fp, txt)
    log(log_fp, 'done.')


# Perform coupled training
def train_coupled(networks, datasets, train_funcs, test_funcs, run_funcs, 
                  log_path):
    lr = 1e-6
    pc_lr = 1e-6
    snapshot_path = os.path.join(log_path, 'snapshots')
    f = open(os.path.join(log_path, 'train_coupled.log'), 'w')
    
    log(f, 'Coupled Training...')
    log(f, 'Learning rates: %f, %f.\nTesting before starting...' % (pc_lr, lr))
    
    _, _, txt = test_scnn(test_funcs, datasets['valid'])
    log(f, 'TEST valid epoch: ' + str(-1) + ' ' + txt)
    
    num_epochs = 30
    mae_history = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        train_switch(networks, datasets, train_funcs, test_funcs, pc_lr, f)
        train_switched_differential(networks, datasets, train_funcs, 
                                    test_funcs, run_funcs, lr, f)
        '''
        if (epoch == 12):
            lr /= 10
            pc_lr /= 10 
            log(f, 'LR CHANGE: %.12f, %.12f.' % (pc_lr, lr))
        '''    
        mae, _, txt = test_scnn(test_funcs, datasets['valid'])
        mae_history[epoch] = mae
        log(f, 'TEST valid epoch: ' + str(epoch) + ' ' + txt)
        
        trained_model_files = [os.path.join(snapshot_path, 
                                    net.name + '_' + str(epoch) + '.pkl') \
                                    for net in networks]
        save_nets(trained_model_files, networks)
        
        p1, = plt.plot(mae_history[: epoch + 1])
        plt.savefig(os.path.join(log_path, 'scnn_mae_history.jpg'))
        plt.clf()
        plt.close()

    min_epoch = np.argmin(mae_history)
    log(f, 'Done Training.\n Minimum loss %f at epoch %d.' % \
                            (mae_history[min_epoch], min_epoch))
    log(f, '\nLoading min epoch...')
    trained_model_files = [os.path.join(snapshot_path,
                                net.name + '_' + str(min_epoch) + '.pkl') \
                                for net in networks]
    load_nets(trained_model_files, networks)
    
    log(f, '\nTesting valid at epoch %d...' % (min_epoch))
    _, _, txt = test_scnn(test_funcs, datasets['valid'])
    log(f, 'epoch: ' + str(min_epoch) + ' ' + txt)
    
    log(f, '\nTesting at epoch %d...' % (min_epoch))
    _, _, txt = test_scnn(test_funcs, datasets['test'])
    log(f, 'epoch: ' + str(min_epoch) + ' ' + txt)
    
    print 'Saving best models...'
    trained_model_files = [os.path.join(log_path, net.name + '.pkl') \
                                for net in networks]
    save_nets(trained_model_files, networks)
    
    log(f, 'done.')
    f.close()
    return


# Load models from differential training and perform coupled training
def train():
    train_images_path = '../dataset/train/images'
    train_gt_path = '../dataset/train/gt'
    test_images_path = '../dataset/test/images'
    test_gt_path = '../dataset/test/gt'
    valid_images_path = '../dataset/valid/images'
    valid_gt_path = '../dataset/valid/gt'
    model_save_path = './models'
    trained_model_files =   [
                            './models/differential_train/deep_patch_classifier.pkl',
                            './models/differential_train/shallow_9x9.pkl',
                            './models/differential_train/shallow_7x7.pkl',
                            './models/differential_train/shallow_5x5.pkl'
                            ]
    
    datasets =  {
                    'train': image_data_set(train_images_path, train_gt_path,
                                            do_shuffle = True),
                    'test':  image_data_set(test_images_path, test_gt_path),
                    'valid': image_data_set(valid_images_path, valid_gt_path)
                }
    networks =  [
                    deep_patch_classifier(),
                    shallow_net_9x9(), 
                    shallow_net_7x7(), 
                    shallow_net_5x5()
                ]
    
    load_nets(trained_model_files, networks)
    train_funcs, test_funcs, run_funcs = create_network_functions(networks)

    path = os.path.join(model_save_path, 'coupled_train')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'snapshots'))
    train_coupled(networks, datasets, train_funcs, test_funcs, run_funcs, path)

    print('\n-------\nDONE.')    
        

if __name__ == '__main__':
    np.random.seed(11)
    train()
    

