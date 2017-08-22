import pandas as pd
from torch import Tensor


class_to_idx = dict()
idx_to_class = dict()
df = pd.read_csv("train_v2_.csv")

fname_to_label_tensor = dict()
def process():
    for i in range(len(df)):
        tags = df.loc[i, 'tags'].split()
        label = [0] * 17
        for tag in tags:
            if tag not in class_to_idx:
                size = len(class_to_idx)
                class_to_idx[tag] = size
                idx_to_class[size] = tag
            label[class_to_idx[tag]] = 1
        fname_to_label_tensor[df.loc[i, 'image_name'] + '.jpg'] = Tensor(label)


process()
