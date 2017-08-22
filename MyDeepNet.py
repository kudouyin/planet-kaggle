import torch
import torchvision
from torch import Tensor
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
from Pre_process import class_to_idx, idx_to_class, process, fname_to_label_tensor

class Deep_Net(nn.Module):
    def extract_feature_0(self, m, i, o):
        feature = F.avg_pool2d(o, o.size(3))
        print('QQQ', feature.view(feature.size(0), -1))

    def extract_feature(self, m, i, o):
        '''
        feature = F.avg_pool2d(o, o.size(3))
        if self.features is not None:
            #print 'self.feature: ' ,self.features.size(), feature.view(feature.size(0), -1).size()
            self.features = torch.cat((self.features, feature.view(feature.size(0), -1)), dim=1)
        else:
            self.features = feature.view(feature.size(0), -1)
        #print 'cnt: ', self.cnt, self.features.size()
        '''
        self.features = i[0]
        #print(self.features)

    def get_model_info_by_name(self, model_name):
        if model_name == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
            fc_num = 2048
        elif model_name == "resnet152":
            model = torchvision.models.resnet152(pretrained=True)
            fc_num = 2048
        elif model_name == "densenet161":
            model = torchvision.models.densenet161(pretrained=True)
            fc_num = 2208
        elif model_name == "vgg19":
            model = torchvision.models.vgg19(pretrained=True)
            fc_num = 25088
        elif model_name == "inception":
            model = torchvision.models.inception_v3(pretrained=True)
            fc_num = 2048
        return model, fc_num

    def __init__(self, model_name):
        super(Deep_Net, self).__init__()
        model, fc_num = self.get_model_info_by_name(model_name)
        self.model = model
        self.features = None
        '''
        self.model.layer1.register_forward_hook(self.extract_feature)
        self.model.layer2.register_forward_hook(self.extract_feature)
        self.model.layer3.register_forward_hook(self.extract_feature)
        self.model.layer4.register_forward_hook(self.extract_feature)
        '''
        if model_name == "resnet50" or model_name == "resnet152" or model_name == "inception":
            #self.model.avgpool.register_forward_hook(self.extract_feature_0)
            self.model.fc.register_forward_hook(self.extract_feature)
        elif model_name == "densenet161" or model_name == "vgg19":
            self.model.classifier.register_forward_hook(self.extract_feature)
        self.fc = nn.Linear(fc_num, 17)

    def forward(self, x):
        self.features = None
        x = self.model(x)
        #print self.feature1.size(), self.feature2.size(),self.feature3.size(),self.feature4.size()
        #x = torch.cat((self.feature1, self.feature2, self.feature3, self.feature4), dim=1)
        #print self.features.size()
        return self.fc(self.features)


    def freeze(self):
        '''
        fc_params_id = list(map(id, self.fc.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, self.parameters())
        for param in base_params:
            param.requires_grad = False
        '''
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_part(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.avgpool.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
