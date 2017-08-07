# test_scnn.py
# Functions to test SCNN
# NOTE: Run this file to test SCNN on trained model.


import numpy as np

from data_reader import image_data_set
from networks import *
from utils import *


# Function to test SCNN on 'dataset'
# NOTE: Testing is done with batch size 1.
# NOTE: 'dataset' MUST yield data such that patches of same 
#       images are together.
def test_scnn(test_funcs, dataset, do_switching = True):
    num_patches = 9
    
    num_subnet = len(test_funcs) - 1
    losses = np.zeros(num_subnet + 2)
    count_losses = np.zeros_like(losses)
    mae_image = np.zeros_like(losses)
    switch_stat = np.zeros(num_subnet)
    pc_switch_stat = np.zeros(num_subnet)
    pc_switch_error = 0.0

    patch_count_accumulator_tmp = np.zeros((num_subnet + 2, 2))
    loss_tmp = np.zeros(num_subnet)
    count_tmp = np.zeros(num_subnet)
    patch_tmp = np.zeros((num_subnet, 2))
    patch_count = 0
    image_count = 0

    for i, (X, Y) in enumerate(dataset):
        for j, test_fn in enumerate(test_funcs[1: ]):
            loss_tmp[j], Y_subnet = test_fn(X, Y)
            patch_tmp[j, 0] = np.sum(Y_subnet)
            patch_tmp[j, 1] = np.sum(Y)
            patch_count_accumulator_tmp[1 + j, 0] += patch_tmp[j, 0] 
            patch_count_accumulator_tmp[1 + j, 1] += patch_tmp[j, 1] 
            count_tmp[j] = np.abs(patch_tmp[j, 0] - patch_tmp[j, 1])
            
        Y_pc = np.argmin(count_tmp)
        losses[-1] += loss_tmp[Y_pc]
        count_losses[-1] += count_tmp[Y_pc]
        patch_count_accumulator_tmp[-1, 0] += patch_tmp[Y_pc, 0]
        patch_count_accumulator_tmp[-1, 1] += patch_tmp[Y_pc, 1]

        # Switching
        if do_switching:
            switch_stat[Y_pc] += 1
            pc_loss, Y_pc_pred = test_funcs[0](X, np.array([Y_pc]))
            Y_pc_pred = np.argmax(Y_pc_pred, axis = 1)[0]
            pc_switch_stat[Y_pc_pred] += 1
            if Y_pc_pred != Y_pc:
                pc_switch_error += 1.0
            losses[0] += pc_loss
            patch_count_accumulator_tmp[0, 0] += patch_tmp[Y_pc_pred, 0]
            patch_count_accumulator_tmp[0, 1] += patch_tmp[Y_pc_pred, 1]

        losses[1: -1] += loss_tmp
        count_losses[1: -1] += count_tmp
        patch_count += 1

        # Compute MAE
        if patch_count >= num_patches:
            patch_count = 0
            mae_image += np.abs(patch_count_accumulator_tmp[:, 0] - \
                                patch_count_accumulator_tmp[:, 1])
            patch_count_accumulator_tmp[:, :] = 0
            image_count += 1

    assert(patch_count == 0)
    i += 1
    print i, image_count
    losses /= i
    count_losses /= i
    switch_stat /= i
    pc_switch_stat /= i
    pc_switch_error /= i
    mae_image /= image_count

    # Printing Results
    txt = ''
    loss_names = ['SCNN', 'R1', 'R2', 'R3', 'SCNN_ideal']
    if not do_switching:
        loss_names = loss_names[1: ]
        count_losses = count_losses[1: ]
        losses = losses[1: ]
        mae_image = mae_image[1: ]
    txt += '\n\tPer_Patch mae [l2 loss]: '
    for i, j, k in zip(loss_names, count_losses, losses):
        txt += ('%s: %.12f [%.12f], ' % (i, j, k))
    txt += '\n\tMAE_Stat: '
    for i, j in zip(loss_names, mae_image):
        txt += ('%s: %.12f, ' % (i, j))
    if do_switching:
        txt += '\n\tSwitch_Stat: '
        for i, j, k in zip(loss_names[1: -1], pc_switch_stat, switch_stat):
            txt += ('%s: %f%% [%f%%], ' % (i, j * 100.0, k * 100.0))
        txt += '\n\tSwitch_Error: %f%%' % (pc_switch_error * 100.0)
        return mae_image[0], mae_image[-1], txt
    return 0.0, mae_image[-1], txt


# Test SCNN on trained model
if __name__ == '__main__':
    test_images_path = '../dataset/test/images'
    test_gt_path = '../dataset/test/gt'
    trained_model_files =   [
                            './models/coupled_train/deep_patch_classifier.pkl',
                            './models/coupled_train/shallow_9x9.pkl',
                            './models/coupled_train/shallow_7x7.pkl',
                            './models/coupled_train/shallow_5x5.pkl'
                            ]
    
    datasets =  {
                    'test':  image_data_set(test_images_path, test_gt_path)
                }
    networks =  [
                    deep_patch_classifier(),
                    shallow_net_9x9(), 
                    shallow_net_7x7(), 
                    shallow_net_5x5()
                ]
    
    load_nets(trained_model_files, networks)
    train_funcs, test_funcs, run_funcs = create_network_functions(networks)
    
    print 'TESTING SCNN...'
    _, _, txt = test_scnn(test_funcs, datasets['test'])
    print txt

    print('\n-------\nDONE.')


