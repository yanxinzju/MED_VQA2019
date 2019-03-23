from model.dataset import med_data
import torch.nn as nn
import torch
from model.util import *
import config
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.lstm=nn.LSTM(300,config.lstm_hidden_size,batch_first=True)

    def forward(self, sentence,sent_len):
        sent_len,idx_sort=np.sort(sent_len)[::-1],np.argsort(-sent_len)
        idx_unsort=np.argsort(idx_sort)
        idx_sort = torch.tensor(idx_sort).to(config.device)
        sentence=sentence.index_select(0,idx_sort)
        sent_packed=nn.utils.rnn.pack_padded_sequence(sentence,sent_len,batch_first=True)
        sent_out,(hn,cn)=self.lstm(sent_packed)

        idx_unsort=torch.tensor(idx_unsort).to(config.device)
        hn=torch.unsqueeze(hn[0].index_select(0,idx_unsort),0)
        cn=torch.unsqueeze(cn[0].index_select(0,idx_unsort),0)

        return hn,cn


class Modality(nn.Module):
    def __init__(self,vocab2int,pretrained_embed,num_class):
        super(Modality, self).__init__()
        self.vocab2int=vocab2int
        self.fc1=nn.Linear(512*7*7,4096)
        self.relu1=nn.ReLU(True)
        self.fc2=nn.Linear(4096,1024)
        self.fc3=nn.Linear(2048,num_class)
        self.embedding=nn.Embedding.from_pretrained(torch.tensor(pretrained_embed))
        self.encoder=Encoder()

    def ques2id(self,question):
        question_int=ques2num(question,self.vocab2int)
        ques_int,ques_len=question_padding(question_int,self.vocab2int['<PAD>'])
        return ques_int,ques_len
    def forward(self, question,image_feature):
        ques_int,ques_len=self.ques2id(question)
        ques_embedding=self.embedding(torch.tensor(ques_int).to(config.device))
        image_feature=torch.tensor(image_feature).to(config.device)
        image_feature=image_feature.view(image_feature.size(0),-1)
        image_feature=self.fc1(image_feature)
        image_feature=self.relu1(image_feature)
        image_feature=self.fc2(image_feature)
        hn,_=self.encoder(ques_embedding.float(),np.array(ques_len))
        hn=hn.squeeze(0).float()
        encode=torch.cat([image_feature,hn],1)
        predict=self.fc3(encode)
        return predict


