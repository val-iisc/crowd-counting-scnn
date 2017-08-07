# make_validation_set.py
# Create validation set
# NOTE: Run this after 'create_dataset.m'.


import os
import random
import shutil


N = 22
images_count = 300
images_path = './train/images'
gt_path = './train/gt'
new_ds_path = './valid'

path = os.path.join(new_ds_path, 'images')
if not os.path.exists(path):
    os.makedirs(path)
p1 = path
path = os.path.join(new_ds_path, 'gt')
if not os.path.exists(path):
    os.makedirs(path)
p2 = path

random.seed(11)
image_ids = random.sample(range(1, images_count + 1), N)

for id in image_ids:
    for i in range(1, 9 + 1):
        f = 'IMG_' + str(id) + '_' + str(i) + '.jpg'
        shutil.move(os.path.join(images_path, f), os.path.join(p1, f))
        f2 = os.path.splitext(f)[0] + '.mat'
        shutil.move(os.path.join(gt_path, f2), os.path.join(p2, f2))

print 'DONE.'


