# -*- coding: utf-8 -*-

import os.path

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

import numpy as np
from PIL import Image

features_dir = './'

img_path = "../data/train/images/synpic371.jpg"
file_name = img_path.split('/')[-1]
feature_path = "test3feat.txt"

transform1 = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)

img = Image.open(img_path)
img1 = transform1(img)

# resnet18 = models.resnet18(pretrained = True)
resnet50_feature_extractor = models.resnet152(pretrained=True)
resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
torch.nn.init.eye(resnet50_feature_extractor.fc.weight)

for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False
# resnet152 = models.resnet152(pretrained = True)
# densenet201 = models.densenet201(pretrained = True)
x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
# y1 = resnet18(x)
y = resnet50_feature_extractor(x)
y = y.data.numpy()
print(y.shape)
np.savetxt(feature_path, y, delimiter=',\n')
# y3 = resnet152(x)
# y4 = densenet201(x)
#
# y_ = np.loadtxt(feature_path, delimiter=',\n').reshape(1, 2048)