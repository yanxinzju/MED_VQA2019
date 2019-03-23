import os
import torch
os.environ['CUDA_VISIABLE_DEVICES']='1'
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
batch_size=32
val_size=1
lstm_hidden_size=1024
epoch=20
lr=1e-4