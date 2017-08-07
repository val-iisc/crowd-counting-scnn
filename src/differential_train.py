# differential_train.py
# Functions to perform differential training
# NOTE: Run this file to do differential training with pretrained weights.


import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_reader import image_data_set
from networks import *
from utils import *
from test_scnn import test_scnn


# Perform differential training
def train_differential(networks, datasets, train_funcs, test_funcs, log_path):
    lr = 1e-6
    num_epochs = 100

    f = open(os.path.join(log_path, 'train_differential.log'), 'w')
    snapshot_path = os.path.join(log_path, 'snapshots')
    log(f, 'Differential Training...')
    log(f, 'Learning rate: %f.\nTesting before starting...' % lr)
    
    _, min_mae, txt = test_scnn(test_funcs, datasets['valid'], False)
    log(f, 'TEST valid epoch: ' + str(-1) + ' ' + txt)

    num_nets = len(test_funcs) - 1
    net_count_losses = np.zeros(num_nets)
    min_mae_history = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        switch_stat = np.zeros(num_nets)
        for i, (X, Y) in enumerate(datasets['train']):

            # Evaluate all regressors.
            for j, (net, test_fn) in \
                    enumerate(zip(networks[1: ], test_funcs[1: ])):
                _, Y_pred = test_fn(X, Y)
                net_count_losses[j] = np.abs(np.sum(Y_pred) - np.sum(Y))
            
            # Backpropagate the regressor with minimum count loss.
            Y_pc = np.argmin(net_count_losses)
            loss = train_funcs[Y_pc + 1](X, Y, lr)
            switch_stat[Y_pc] += 1
            if i % 200 == 0 and i > 0:
                txt = ''
                for j in range(switch_stat.shape[0]):
                    txt += 'n%d: %d, ' % (j, switch_stat[j])
                log(f, txt)
        txt = ''
        for i in range(switch_stat.shape[0]):
            txt += 'n%d: %d, ' % (i, switch_stat[i])
        log(f, 'epoch: ' + str(epoch) + ' ' + txt)
        
        _, min_mae, txt = test_scnn(test_funcs, datasets['valid'], False)
        log(f, 'TEST valid epoch: ' + str(epoch) + ' ' + txt)
        min_mae_history[epoch] = min_mae
        diff_trained_model_files = [os.path.join(snapshot_path, 
                                            net.name + '_' + str(epoch) + '.pkl') \
                                        for net in networks[1: ]]
        save_nets(diff_trained_model_files, networks[1: ])
        '''
        if epoch == 44:
            lr = 1e-7
            log(f, 'LR CHANGE Learning rate: %f.' % lr)
        '''
        p1, = plt.plot(min_mae_history[: epoch + 1])
        plt.savefig(os.path.join(log_path, 'min_mae_history.jpg'))
        plt.clf()
        plt.close()

    '''    
    min_epoch = np.argmin(min_mae_history)
    min_val = np.min(min_mae_history)
    log(f, 'Minimum at epoch: ' + str(min_epoch) + ' (' + str(min_val) + ')')
    diff_trained_model_files = [os.path.join(snapshot_path,
                                    net.name + '_' + str(min_epoch) + '.pkl') \
                                    for net in networks[1: ]]
    load_nets(diff_trained_model_files, networks[1: ])
    _, _, txt = test_scnn(test_funcs, datasets['valid'], False)
    log(f, 'TEST valid at minimum epoch: ' + str(epoch) + ' ' + txt)
    '''

    print 'Saving best models...'
    diff_trained_model_files = [os.path.join(log_path, net.name + '.pkl') \
                                for net in networks]
    save_nets(diff_trained_model_files, networks)
    
    log(f, 'Done.')
    f.close()
    return


# Load pretrained models and perform differential training
def train():
    train_images_path = '../dataset/train/images'
    train_gt_path = '../dataset/train/gt'
    test_images_path = '../dataset/test/images'
    test_gt_path = '../dataset/test/gt'
    valid_images_path = '../dataset/valid/images'
    valid_gt_path = '../dataset/valid/gt'
    model_save_path = './models'
    trained_model_files =   [
                            './models/pretrain/switch_classifier.pkl',
                            './models/pretrain/9x9_net.pkl',
                            './models/pretrain/7x7_net.pkl',
                            './models/pretrain/5x5_net.pkl'
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

    path = os.path.join(model_save_path, 'differential_train')
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'snapshots'))
    train_differential(networks, datasets, train_funcs, test_funcs, path)

    print('\n-------\nDONE.')    
        

if __name__ == '__main__':
    np.random.seed(11)
    train()
    
    
