from __future__ import print_function
import os
import pandas as pd
import torch
import torchvision
from torch import Tensor
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import argparse
import sys
import time
import random
import pickle

from Pre_process import class_to_idx, idx_to_class, process, fname_to_label_tensor

class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

def get_mytransforms(type):
    if type == -1:
        transformations = None
    if type == 0:
        transformations = transforms.Compose([\
                                          transforms.Scale(224+5),\
                                          transforms.RandomCrop(224),\
                                          transforms.RandomHorizontalFlip(),\
                                          ])
    elif type == 1:
        transformations = transforms.Compose([\
                                          transforms.Scale(224+5),\
                                          transforms.RandomCrop(224),\
                                          RandomVerticalFlip(),\
                                          ])
    elif type == 2:
        transformations = transforms.Compose([\
                                          transforms.Scale(224+5),\
                                          transforms.RandomCrop(224),\
                                          transforms.RandomHorizontalFlip(),\
                                          RandomVerticalFlip(),\
                                          ])
    if type == 3:
        transformations = transforms.Compose([\
                                          transforms.Scale(256+5),\
                                          transforms.RandomCrop(256),\
                                          transforms.RandomHorizontalFlip(),\
                                          ])
    elif type == 4:
        transformations = transforms.Compose([\
                                          transforms.Scale(256+5),\
                                          transforms.RandomCrop(256),\
                                          RandomVerticalFlip(),\
                                          ])
    elif type == 5:
        transformations = transforms.Compose([\
                                          transforms.Scale(256+5),\
                                          transforms.RandomCrop(256),\
                                          transforms.RandomHorizontalFlip(),\
                                          RandomVerticalFlip(),\
                                          ])
    if type == 6:
        transformations = transforms.Compose([\
                                          transforms.Scale(320+5),\
                                          transforms.RandomCrop(320),\
                                          transforms.RandomHorizontalFlip(),\
                                          ])
    elif type == 7:
        transformations = transforms.Compose([\
                                          transforms.Scale(320+5),\
                                          transforms.RandomCrop(320),\
                                          RandomVerticalFlip(),\
                                          ])
    elif type == 8:
        transformations = transforms.Compose([\
                                          transforms.Scale(320+5),\
                                          transforms.RandomCrop(320),\
                                          transforms.RandomHorizontalFlip(),\
                                          RandomVerticalFlip(),\
                                          ])
    return transformations

def show_time(sec):
    sec_int = int(sec)
    hour = sec_int / 3600
    minute = (sec_int - hour*3600) / 60
    seconds = sec_int - hour * 3600 - minute * 60
    mseconds = int((sec- sec_int)*1000)
    ret = ""
    if hour != 0:
        ret += str(hour) + 'h'
    if minute != 0:
        ret += str(minute) + 'm'
    if seconds != 0:
        ret += str(seconds) + 's'
    else:
        ret += str(mseconds) + 'ms'
    return ret

def my_scorer(truth, pred):
    #print truth.sum(), pred
    SMALL = 1e-12
    p = pred > 0.2
    num_pos = p.sum() + SMALL
    num_pos_hat = truth.sum()

    tp = (truth * p).sum()

    #print p, tp, num_pos, num_pos_hat
    precise     = tp/num_pos
    recall      = tp/num_pos_hat

    f2 = (1+4)*precise*recall/(4 * precise + recall + SMALL)
    return f2

def get_random_split_file_list(root, index, Fold = 10):
    '''
    random.seed(index)
    file_list = sorted(os.listdir(root))
    train_list = sorted(random.sample(file_list, int(len(file_list) * 0.92)))
    val_list = sorted(list(set(file_list) - set(train_list)))
    return train_list, val_list
    '''
    random.seed(0)
    file_list = os.listdir(root)
    random.shuffle(file_list)
    length = len(file_list)
    length_fold = int(length/Fold) + 1

    st = index * length_fold
    end = min(length, st + length_fold)
    val_list = sorted(file_list[st:end])
    train_list = sorted(list(set(file_list) - set(val_list)))
    return train_list, val_list

def make_dataset_generator(root, batch_size, file_list=None, my_transforms=None):
    imgset = []
    labelset = []
    fcnt = 1
    if file_list is None:
        file_list = sorted(os.listdir(root))
    for fname in file_list:
        path = os.path.join(root, fname)
        img = Image.open(path)
        if my_transforms is not None:
            img_tensor = my_transforms(img.convert('RGB'))
            img_tensor = transforms.ToTensor()(img_tensor)
        else:
            img_tensor = transforms.ToTensor()(img.convert('RGB'))
        if len(imgset) > 0 and imgset[-1].shape != img_tensor.numpy().shape:
            imgset = np.array(imgset)
            labelset = np.array(labelset)
            yield imgset, labelset
            imgset = []
            labelset = []
        imgset.append(img_tensor.numpy())

        if fname in fname_to_label_tensor.keys():
            label_tensor = fname_to_label_tensor[fname]
            #print(label_tensor)
            labelset.append(label_tensor.numpy())
        else:
            label_tensor = [None]*17      # this is predict mode, so make it -1, and won't be used
            labelset.append(label_tensor)
        #print('size: ', len(imgset))
        if fcnt % batch_size == 0 or fcnt == len(file_list):
            #print(imgset)
            imgset = np.array(imgset)
            labelset = np.array(labelset)
            yield imgset, labelset
            imgset = []
            labelset = []
        fcnt += 1

def gettagbypred(Y_preds):
    Y_preds_tags = []
    for raw in Y_preds:
        tags = []
        for index in range(len(raw)):
            if raw[index] > 0.2:
                tags.append(idx_to_class[index])
        Y_preds_tags.append(' '.join(tags))
    return np.array(Y_preds_tags)

if __name__ == "__main__":
    f1, f2 = get_random_split_file_list('train-jpg', 0)
    print(len(f2))
    f1, f2 = get_random_split_file_list('train-jpg', 9)
    print(len(f2))
