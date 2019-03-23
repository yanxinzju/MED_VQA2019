import sys
sys.path.append('..')
import config
import tables
import torch
import os
import torch.nn as nn
from model.basic_model import Modality
from model.dataset import med_data
from torch.utils.data import DataLoader
from model.util import *

train_vgg_feature = tables.open_file('../file/train_vgg16_feature.h5').root.vgg16
val_vgg_feature = tables.open_file('../file/val_vgg16_feature.h5').root.vgg16
# def Abnormality_train():
#     train_data=med_data()



def Modality_train(pretrained=False):
    train_data=med_data('Modality.csv',mode='train')
    train_loader=DataLoader(dataset=train_data,batch_size=config.batch_size,shuffle=True,drop_last=True)

    vocab2int=load_json('Modality','vocab2int')
    embedding=np.load('../file/Modality/embedding.npy')
    name2id=load_name2id(mode='train')
    answer_set=load_answer_set('Modality')

    num_class = len(answer_set)

    model = Modality(vocab2int, embedding, num_class).to(config.device)

    if pretrained==False:
        ckpt_id=0
    else:
        ckpt_id=load_last_ckpt('Modality')
        print('load pretrained model at ckpt_{}'.format(ckpt_id))
        model.load_state_dict(torch.load('../log/Modality/' + 'ckpt_' + str(ckpt_id) + '.pkl'))
        ckpt_id+=1

    criterion=nn.CrossEntropyLoss()

    optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)

    for epoch in range(config.epoch):
        j=0
        total_loss=0
        total_acc=0
        for step,sample in enumerate(train_loader):
            optimizer.zero_grad()
            j+=1
            image_name=sample['image_name']
            question=sample['question']
            answer=torch.tensor(answer2num(sample['answer'],answer_set)).to(config.device)
            image_feature=[]
            for i in range(config.batch_size):
                feature=train_vgg_feature[name2id[image_name[i]]]
                image_feature.append(feature)
            out=model(question,image_feature)
            loss=criterion(out,answer)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            predict=torch.argmax(out,1)
            acc=torch.sum(predict==answer)
            total_acc+=acc.item()
            if j%20==0:
                print('[{}/{}],train loss=: {},train acc=: {}'.format(j,train_data.__len__()/config.batch_size,total_loss/j,total_acc/(j*config.batch_size)))
        total_loss=total_loss/j
        total_acc=total_acc/train_data.__len__()
        print('\n')
        print('train epoch={}/{},train loss=: {},train acc=: {}'.format(epoch+ckpt_id,config.epoch+ckpt_id,total_loss,total_acc))
        print('\n')
        torch.save(model.state_dict(),'../log/Modality/'+'ckpt_'+str(epoch+ckpt_id)+'.pkl')
        Modality_val(epoch+ckpt_id)


def Modality_val(ckpt_id):
    val_data=med_data('Modality.csv',mode='val')
    val_loader=DataLoader(dataset=val_data,batch_size=config.val_size,shuffle=False,drop_last=True)

    vocab2int=load_json('Modality','vocab2int')
    embedding=np.load('../file/Modality/embedding.npy')
    name2id=load_name2id(mode='val')
    answer_set=load_answer_set('Modality')

    num_class = len(answer_set)

    model = Modality(vocab2int, embedding, num_class).to(config.device)

    criterion = nn.CrossEntropyLoss()

    print('load model at ckpt_{}'.format(ckpt_id))
    model.load_state_dict(torch.load('../log/Modality/' + 'ckpt_' + str(ckpt_id) + '.pkl'))

    total_loss=0
    total_acc=0
    for step,sample in enumerate(val_loader):
        image_name=sample['image_name']
        question=sample['question']
        answer=torch.tensor(answer2num(sample['answer'],answer_set)).to(config.device)
        image_feature=[]
        for i in range(config.val_size):
            feature=val_vgg_feature[name2id[image_name[i]]]
            image_feature.append(feature)
        out=model(question,image_feature)
        loss=criterion(out,answer)
        predict=torch.argmax(out,1)
        # print(predict)
        # print(answer)
        # print('\n')
        acc=torch.sum(predict==answer)
        total_acc+=acc.item()
        total_loss+=loss.item()
    total_loss=total_loss/val_data.__len__()
    total_acc=total_acc/val_data.__len__()
    print('val loss=: {},val acc=: {}'.format(total_loss,total_acc))
    print('\n')
    print('\n')

if __name__ == '__main__':
    Modality_train(pretrained=True)
    # Modality_val(19)
