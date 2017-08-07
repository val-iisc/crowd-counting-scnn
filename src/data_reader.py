# data_reader.py
# Functions to read data
# NOTE: Before using 'data_reader' functions, the dataset MUST be created
#       with 'dataset/create_datasets.m'.
# NOTE: 'data_reader' assumes that images and corresponding density maps
#       are available as patches. Patches of the same image MUST be named
#       with suffix '_x', where 'x' stands for patch id starting from
#       1 to num_patches (= 9). Use 'dataset/create_datasets.m' to do this.
#       Eg: Patches of IMG_100.jpg should be IMG_100_1.jpg, IMG_100_2.jpg,...
#           with corresponding density maps as IMG_100_1.mat, IMG_100_2.mat,...


import numpy as np
import cv2
import os
import random
import scipy.io
import theano


# Data reader class
class image_data_set():
    
    # 'images_path': Path for image patches.
    # 'gt_path': Path for corresponding density maps.
    # 'do_shuffle': Whether to shuffle dataset.
    def __init__(self, images_path, gt_path, do_shuffle = False):
        self.images_path = images_path
        self.gt_path = gt_path
        self.image_files = [f \
                            for f in os.listdir(images_path) \
                            if os.path.isfile(os.path.join(images_path, f))]
        assert(len(self.image_files) > 0)
        if do_shuffle:
            random.seed(11)
        else:
            print 'TEST dataset sorting files...'
            self.image_files.sort()
            print 'Done'
        self.num_examples = len(self.image_files)
        self.do_shuffle = do_shuffle
        
    def __iter__(self):
        if self.do_shuffle:
            print 'TRAIN so random shuffle....'
            random.shuffle(self.image_files)
            files = self.image_files
        else:
            print 'TEST...'
            files = self.image_files
             
        for f in files:
            image = cv2.imread(os.path.join(self.images_path, f))
            if image is None:
                print 'Unable to read image %s. Exiting...' % f
                exit(0)
            if len(image.shape) > 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            X = image.reshape((1, 1, image.shape[0], image.shape[1]))
            X = X.astype(theano.config.floatX)
            data_mat = scipy.io.loadmat(os.path.join(self.gt_path,
                                    os.path.splitext(f)[0] + '.mat'))
            Y = data_mat['final_gt'].astype(theano.config.floatX)
            Y = Y.reshape((1, 1, Y.shape[0], Y.shape[1]))
            yield (X, Y)


