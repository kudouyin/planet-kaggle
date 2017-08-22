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
import gc

from Myutil import get_mytransforms, show_time, my_scorer, get_random_split_file_list, make_dataset_generator, gettagbypred

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--small', action='store_true', help='use small data')
parser.add_argument('--aug_type', type=int, default = 0, help='train with aug type')
parser.add_argument('--freeze', type=str, default='other', help='freeze model')
parser.add_argument('--bs', type=int, default=64, help='input batch size')
parser.add_argument('--sleep_epoch', type=int, default=0, help='sleep epoch')
parser.add_argument('--sleep_seconds', type=int, default=1200, help='sleep seconds')
parser.add_argument('--dif_lr', type=str, default="False", help='choose the optimizer')
parser.add_argument('--epoch', type=int, default=500, help='input epoch ')
parser.add_argument('--st_epoch', type=int, default=0, help='input start epoch ')
parser.add_argument('--end_epoch', type=int, default=10, help='input end epoch ')
parser.add_argument('--lr', type=float, default=0.0001, help='input learning_rate')
parser.add_argument('--test', action='store_true', help='predict mode')
parser.add_argument('--model', type=str, default='resnet50', help='model name')
args = parser.parse_args()

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2

def train(model, train_set_generator, optimizer, train_files_length ):
    time1 = time.time()
    model.train()
    train_loss = 0
    sample_index = 0
    output_all = None
    label_all = None

    for item in train_set_generator:
        model.zero_grad()      #if not clean grad, train will fail
        data, label = Variable(Tensor(item[0])), Variable(Tensor(item[1]))
        sample_index += len(data)
        if torch.cuda.is_available() and args.cuda:
            data, label = data.cuda(), label.cuda()
        output = model(data)
        error = criterion(output, label)
        train_loss += error.data[0] * len(data)
        if output_all is None:
            output_all = F.sigmoid(output).cpu().data.numpy()
            label_all = label.cpu().data.numpy()
        else:
            output_all = np.concatenate((output_all, F.sigmoid(output).cpu().data.numpy()), axis=0)
            label_all = np.concatenate((label_all, label.cpu().data.numpy()), axis=0)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        sys.stdout.write("\t%d/%d ---loss: %.4f\r"%(sample_index, train_files_length, train_loss))
        sys.stdout.flush()
        time.sleep(0.5)
    f2_score = my_scorer(output_all, label_all)
    time2 = time.time()
    print("\ttrain_loss: %.4f f2: %.4f train_time: %s"%(train_loss/sample_index, f2_score, show_time(time2-time1)))

def validation(model, validate_set_generator):
    model.eval()
    test_loss = 0
    time1 = time.time()
    output_all = None
    label_all = None
    sample_index = 0
    for item in validate_set_generator:
        data, label = Variable(Tensor(item[0]), volatile=True), Variable(Tensor(item[1]))
        sample_index += len(data)
        if torch.cuda.is_available() and args.cuda:
            data, label = data.cuda(), label.cuda()
        output = model(data)
        if output_all is None:
            output_all = F.sigmoid(output).cpu().data.numpy()
            label_all = label.cpu().data.numpy()
        else:
            output_all = np.concatenate((output_all, F.sigmoid(output).cpu().data.numpy()), axis=0)
            label_all = np.concatenate((label_all, label.cpu().data.numpy()), axis=0)
        #print "sig output: ", F.sigmoid(output)
        test_loss += criterion(output, label).data[0] * len(data)

    f2_score = my_scorer(output_all, label_all)
    time2 = time.time()
    print("\tvalidate_set_loss: %.4f f2: %.4f val_time: %s" % (test_loss/sample_index, f2_score, show_time(time2 - time1)))
    return test_loss/sample_index, f2_score

def test(model, test_set_generator):
    model.eval()
    time1 = time.time()
    preds = None
    test_files_length = len(os.listdir(my_test_path))
    test_additional_files_length = len(os.listdir(my_test_path_additional))
    sample_index = 0
    for item in test_set_generator:
        data = Variable(Tensor(item[0]), volatile=True)           #label is None
        sample_index += len(data)
        if torch.cuda.is_available() and args.cuda:
            data = data.cuda()
        output = model(data)
        output = F.sigmoid(output)
        if preds is None:
            preds = output.data.cpu().numpy()
        else:
            preds = np.concatenate((preds, output.data.cpu().numpy()), axis=0)

        sys.stdout.write("\ttest complete %d/%d or %d cases\r"%(sample_index, test_files_length, test_additional_files_length))
        sys.stdout.flush()
        time.sleep(0.5)
    time2 = time.time()
    print("test time: " + show_time(time2 - time1) + ' '*50)
    return preds

import MyDeepNet
criterion = nn.MultiLabelSoftMarginLoss()
if torch.cuda.is_available() and args.cuda:
    criterion.cuda()

patience = 3
train_path = 'train-jpg'
my_test_path = 'test-jpg'
my_test_path_additional = 'test-jpg-additional'

if args.small:
    my_train_path = 'small-train-jpg'
    my_validation_path = 'small-validation-jpg'
    my_test_path = 'small-test-jpg'
    my_test_path_additional = 'small-test-jpg-additional'

def trigger_train(model_name, aug_type, cv=0):
    train_list, val_list = get_random_split_file_list(train_path, cv)
    print("random split data train(%d) val(%d)" % (len(train_list), len(val_list)))
    print("train model(%s) with aug type %d"%(model_name, aug_type))
    run_model = MyDeepNet.Deep_Net(model_name)

    if torch.cuda.is_available() and args.cuda:
        run_model.cuda()

    if os.path.exists(model_name + '_cv' + str(cv) + "_training_loss_best.pt"):
        print('train from saved best loss check point')
        run_model.load_state_dict(torch.load(model_name + '_cv' + str(cv) + '_training_loss_best.pt'))


    best_val_loss = 1 << 30
    best_f2_score = 0
    best_epoch = 0
    cnt = 0
    st_time = time.time()

    print("dif_lr", args.dif_lr)
    if args.freeze == 'all':
        print('just train last layer')
        run_model.freeze()
        optimizer = torch.optim.Adam(run_model.fc.parameters(), lr = args.lr)
    elif args.freeze == 'part':
        run_model.freeze_part()
        print('train last block and fc layers')
        optimizer = torch.optim.Adam([{'params': run_model.model.layer4.parameters()},\
                                    {'params': run_model.model.avgpool.parameters()},\
                                    {'params': run_model.fc.parameters()},\
                                    ], lr = args.lr)
    else:
        print('train all layers')
        run_model.unfreeze()
        if args.dif_lr == "True":
            print("use the different lr")
            ignored_params = list(map(id, run_model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params,
                                 run_model.parameters())

            optimizer_adam = torch.optim.Adam([
                        {'params': base_params},
                        {'params': run_model.fc.parameters(), 'lr': args.lr}
                    ], lr=args.lr*0.1)

            optimizer = optimizer_adam
        else:
            print("use the same lr")
            optimizer = torch.optim.Adam(run_model.parameters(), lr = args.lr)

    for epoch in range(args.epoch):
        print('epoch %d ' % (epoch), end=" ")
        for param_group in optimizer.param_groups:
            print("(lr:%.12f) " % (param_group['lr']))
        if args.sleep_epoch != 0 and (epoch+1) % args.sleep_epoch == 0:
            print('sleeping sleeping!!!!')
            time.sleep(args.sleep_seconds)

        train_set_features_generator = make_dataset_generator(train_path, args.bs, train_list, get_mytransforms(aug_type))
        train(run_model, train_set_features_generator, optimizer, len(train_list))

        validation_set_features_generator = make_dataset_generator(train_path, args.bs, val_list, get_mytransforms(aug_type))
        val_loss, f2_score = validation(run_model, validation_set_features_generator)

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            cnt = 0
            torch.save(run_model.state_dict(), model_name + '_cv' + str(cv) + '_training_loss_best.pt')

        elif val_loss > best_val_loss:
            cnt += 1
            run_model.load_state_dict(torch.load(model_name + '_cv' + str(cv) + '_training_loss_best.pt'))
            print("\tworse loss, so train from best loss check point")

        if epoch >= 1:
            adjust_learning_rate(optimizer)

        if cnt >= patience:
            print("early stop")
            break

    end_time = time.time()
    print("train time: %s" % (show_time(end_time - st_time)))

def trigger_test(model_name, aug_type, cv=0):
    print("predict start (model name:%s)"%(model_name))
    new_model = MyDeepNet.Deep_Net(model_name)
    #new_model = MyUNet.UNet()
    if args.cuda:
        new_model.cuda()
    new_model.load_state_dict(torch.load(model_name + '_cv' + str(cv) + '_training_loss_best.pt'))

    preds1 = []
    aug_types = [0,1,2,3,4,5]

    for i in aug_types:
        print('test aug index: ', i, ' '*50)
        if args.sleep_epoch != 0 and (i+1) % args.sleep_epoch == 0:
            print("test sleep")
            time.sleep(args.sleep_seconds)
        test_set_features_generator = make_dataset_generator(my_test_path, test_bs, my_transforms=get_mytransforms(i))
        preds1_x = test(new_model, test_set_features_generator)
        pickle.dump(preds1_x, open(model_name + '_cv' + str(cv) + "_aug_preds" + str(i) + ".pkl", "wb"))
        preds1.append(preds1_x)
    preds1_base = preds1[aug_type]
    preds1 = np.mean(np.array(preds1), axis = 0)

    preds2 = []
    for i in aug_types:
        print('test additional aug index: ', i, ' '*50)
        if args.sleep_epoch != 0 and (i+1) % args.sleep_epoch == 0:
            print("test additional sleep")
            time.sleep(args.sleep_seconds)
        test_set_additioanal_features_generator = make_dataset_generator(my_test_path_additional, test_bs, my_transforms=get_mytransforms(i))
        preds2_x = test(new_model, test_set_additioanal_features_generator)
        pickle.dump(preds2_x, open(model_name + '_cv' + str(cv) + "_aug_preds_add" + str(i) + ".pkl", "wb"))
        preds2.append(preds2_x)
    preds2_base = preds2[aug_type]
    preds2 = np.mean(np.array(preds2), axis = 0)

    preds_base = np.concatenate((preds1_base, preds2_base), axis = 0)
    preds = np.concatenate((preds1, preds2), axis = 0)
    pickle.dump(preds, open(model_name + '_cv' + str(cv) + "_aug_preds.pkl", "wb"))

    preds_base_tag = gettagbypred(preds_base)
    preds_tag = gettagbypred(preds)
    print(preds_tag)

    file_list1 = sorted(os.listdir('test-jpg'))
    file_list2 = sorted(os.listdir('test-jpg-additional'))
    Y_name = np.array(file_list1 + file_list2)

    answer_base = pd.DataFrame()
    answer_base['image_name'] = [item.split('.')[0] for item in Y_name]
    answer_base['tags'] = preds_base_tag
    answer_base.to_csv('answer_net_base.csv', index=False)

    answer = pd.DataFrame()
    answer['image_name'] = [item.split('.')[0] for item in Y_name]
    answer['tags'] = preds_tag

    print(answer)
    answer.to_csv('answer_net.csv', index=False)

if __name__ == "__main__":
    print("main start")
    test_bs = 224
    if not args.test:
        for index in range(args.st_epoch, args.end_epoch):
            print("cv %d train start"%(index))
            trigger_train(args.model, args.aug_type, index)
            print("cv %d test start"%(index))
            trigger_test(args.model, args.aug_type, index)
    else:
        test_bs = args.bs
        for index in range(args.st_epoch, args.end_epoch):
            print("cv %d test start"%(index))
            trigger_test(args.model, args.aug_type, index)
