import numpy as np
import pandas as pd
import pickle
import os
from Myutil import gettagbypred
import random

def ensemble(pkl_list, output_file_name, fun=None):
    preds = []
    for pkl_file in pkl_list:
        #print(pkl_file)
        if pkl_file not in os.listdir('.'):
            continue
        preds_x = pickle.load(open(pkl_file, "rb"))
        preds.append(preds_x)
    preds = np.array(preds)
    if fun is None:
        preds = np.mean(preds, axis = 0)
    else:
        preds = fun(preds)
    preds_tag = gettagbypred(preds)

    file_list1 = sorted(os.listdir('test-jpg'))
    file_list2 = sorted(os.listdir('test-jpg-additional'))
    Y_name = np.array(file_list1 + file_list2)

    answer = pd.DataFrame()
    answer['image_name'] = [item.split('.')[0] for item in Y_name]
    answer['tags'] = preds_tag
    answer.to_csv(output_file_name, index=False)

def random_get_mean(preds):
    model_num, case_num, class_num = preds.shape
    ret = np.zeros((case_num, class_num))
    for i in range(case_num):
        for j in range(class_num):
            sample = random.sample(range(model_num), int(model_num*0.8))
            sum_x = 0
            for k in sample:
                sum_x += preds[k, i, j]
            ret[i, j] = sum_x / len(sample)
    return ret

if __name__ == "__main__":
    densenet161_pkl_list = []
    for i in range(10):
        pkl_file = "densenet161" + '_cv' + str(i) + "_aug_preds.pkl"
        densenet161_pkl_list.append(pkl_file)
    ensemble(densenet161_pkl_list, "densenet161_answer_net.csv")

    resnet50_pkl_list = []
    for i in [0, 1, 4, 5]:
        pkl_file = "resnet50" + '_cv' + str(i) + "_aug_preds.pkl"
        resnet50_pkl_list.append(pkl_file)
    #ensemble(resnet50_pkl_list, "renet50_answer_net.csv")

    vgg19_pkl_list = []
    for i in [0, 1]:
        pkl_file = "vgg19" + '_cv' + str(i) + "_aug_preds.pkl"
        vgg19_pkl_list.append(pkl_file)


    all_pkl_list = densenet161_pkl_list + resnet50_pkl_list + vgg19_pkl_list
    print(all_pkl_list)
    ensemble(all_pkl_list, "all_answer_net.csv")
