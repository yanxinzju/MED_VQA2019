import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = models.resnet152(pretrained=True)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output


model = net()
# model = model.cuda()


def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)
    print(img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    print(x.shape)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    print(y.shape)
    y = torch.squeeze(y)
    print(y.shape)
    y = y.data.numpy()
    print(y.shape)
    np.savetxt(saved_path, y, delimiter=',')


if __name__ == '__main__':
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    use_gpu = torch.cuda.is_available()

    x_path = '../data/train/images/synpic371.jpg'

    fx_path = "./myfeat.txt"

    extractor(x_path, fx_path, model, use_gpu)