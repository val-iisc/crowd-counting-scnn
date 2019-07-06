### Crowd-Counting-Cnn

>**NOTE: This repository is deprecated and would be removed shortly. Please use the latest code base in https://github.com/val-iisc/lsc-cnn.**

This project is an implementation of the crowd counting model proposed in our CVPR 2017 paper - Switching Convolutional Neural Network([SCNN](https://arxiv.org/abs/1708.00199))for Crowd Counting. SCNN is an adaptation of the fully-convolutional neural network  and uses an expert CNN that chooses the best crowd density CNN regressor for parts of the scene from a bag of regressors. This helps it tackle intra-scene crowd density variation and obtain SOTA results


### License

This code is released under the MIT License (Please refer to the LICENSE file for details).
### Citation
Please cite our paper in your publications if it helps your research:
    
    
    @article{2017arXiv170800199B,
    Author = {Babu Sam, Deepak and Surya, Shiv and 
    Babu R, Venkatesh},
        Title = {Switching Convolutional Neural Network for Crowd Counting},
        Journal = {ArXiv e-prints},
        eprint = {1708.00199},
        Keywords = {Computer Science - Computer Vision and Pattern Recognition},
        Year = {2017},
        Month = {august},
       }
<!---
    @inproceedings{,
        Author = {},
        Title = {},
        Booktitle = {},
        Year = {2016}
    }
--->
### Dependencies and Installation

1. Code for SCNN is based on Lasagne\Theano. This code was tested on UBUNTU 14.04 on the folowing NVIDIA GPUs: NVIDIA TITAN X. 

2. To test SCNN on trained model:
  
   ```bash
   $ git clone https://github.com/val-iisc/crowd-counting-scnn.git
   $ matlab -nodisplay -nojvm -nosplash -nodesktop -r "run('dataset/create_test_set.m');" 
   $ python ./src/test_scnn.py
   ```
3. To train SCNN:
  
   ```bash
   $ git clone https://github.com/val-iisc/crowd-counting-scnn.git
   $ matlab -nodisplay -nojvm -nosplash -nodesktop -r "run('dataset/create_datasets.m');" 
   $ python ./src/differential_train.py
   $ python ./src/coupled_train.py
   ```


### Q&A
Where can we find MCNN create_density.m?
This function and the dataset are not included in this release as we are not owners of the dataset and cannot release it. Please contact authors of the dataset (they are authors of this paper http://ieeexplore.ieee.org/document/7780439/?reload=true) and the code will work fine. You do not need this funciton to benchmark using the trained models that are hosted. The authors of the dataset were prompt and courteous in our communication with them and you should have no trouble as along you use your academic credentials to contact them. We use the same density function to avoid any implementation bias as the density is the supervisory signal for training these models.
