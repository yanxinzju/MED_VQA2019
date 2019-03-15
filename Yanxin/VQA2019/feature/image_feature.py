import torch
import torch.nn as nn
import torchvision
import tables
import cv2
import sys
sys.path.append('..')
from model.util import *

os.environ['CUDA_VISIBLE_DEVICES']='3'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Resnet152(nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        self.resnet152=torchvision.models.resnet152(pretrained=True)
        self.module_list=list(self.resnet152.children())
        self.conv5=nn.Sequential(*self.module_list[:-2])
        self.pool5=self.module_list[-2]
    def forward(self,x):
        # print(x.shape)
        x=x.unsqueeze(0)
        x=x.permute(0,3,1,2)
        x=self.conv5(x)
        x=x.squeeze(0)
        # print(x.shape)
        return x

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.vgg16=torchvision.models.vgg16(pretrained=True).features
    def forward(self, x):
        x = x.unsqueeze(0)
        x=x.permute(0,3,1,2)
        x=self.vgg16(x)
        x = x.squeeze(0)
        return x

def extract_vgg16_feature(mode):
    image_file=load_name2id(mode)
    save_path='../file/'+mode+'_vgg16_feature.h5'
    h5file = tables.open_file(save_path, 'w')
    image_path=os.path.join('../data',mode,'images')
    vgg16=Vgg16().to(device)
    vgg_feature=[]
    for key in image_file:
        print(key)
        img=cv2.imread(os.path.join(image_path,key+'.jpg'))
        img = cv2.resize(img, (224, 224))
        image = torch.tensor(img).float().to(device)
        feature=vgg16(image).cpu().detach().numpy()
        vgg_feature.append(feature)
    h5file.create_array('/','vgg16',vgg_feature,'vgg16 feature')
    h5file.close()


if __name__ == '__main__':
    extract_vgg16_feature()
