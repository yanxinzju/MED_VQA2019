import torch.nn as nn
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer,BertModel
from model.util import *


class Embeddinglayer(nn.Module):
    def __init__(self,input_size,emsize,trainable=True):
        super(Embeddinglayer, self).__init__()
        self.input_size=input_size
        self.emsize=emsize
        self.embedding=nn.Embedding(input_size,emsize)
        if trainable==False:
            self.embedding.weight.requires_grad=False

    def forward(self, input):
        return self.embedding(input)

    def init_embedding_weights(self,int2word,dictionary):
        pretrained_weight=np.empty([self.input_size,self.emsize],dtype=float)
        for i in range(self.input_size):
            try:
                word=int2word[i]
                pretrained_weight[i]=dictionary[word]
            except KeyError:
                pretrained_weight[i]=np.random.normal(scale=0.6,size=(self.emsize,))
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

class LstmLayer(nn.Module):
    def __init__(self,emsize,hidden_size,batch_size,device):
        super(LstmLayer, self).__init__()

        self.emsize=emsize
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.device=device
        self.lstm=nn.LSTM(self.emsize,self.hidden_size,batch_first=True)

    def forward(self, sentence,sent_len,h0,c0):
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort).to(self.device)
        sentence = sentence.index_select(0, idx_sort)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sentence, sent_len, batch_first=True)
        sent_out, (hn, cn) = self.lstm(sent_packed, (h0, c0))

        idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        hn = torch.unsqueeze(hn[0].index_select(0, idx_unsort), 0)
        cn = torch.unsqueeze(cn[0].index_select(0, idx_unsort), 0)

        return hn, cn
    def init_hidden(self):
        h0=torch.zeros(1,self.batch_size,self.hidden_size).to(self.device)
        c0=torch.zeros(1,self.batch_size,self.hidden_size).to(self.device)
        return h0,c0


class GloveEncoder(nn.Module):
    def __init__(self,input_size,emsize,int2vocab,vocab2int,dictionary,device):
        super(GloveEncoder, self).__init__()

        self.input_size=input_size
        self.emsize=emsize
        self.device=device
        self.int2vocab=int2vocab
        self.vocab2int=vocab2int
        self.embedding=Embeddinglayer(self.input_size,self.emsize)
        self.embedding.init_embedding_weights(int2vocab,dictionary)
        # self.encoder=LstmLayer(self.emsize,self.hidden_size,self.batch_size,self.device)
    def ques2id(self,question):
        ques_int = question2num(question, self.vocab2int)
        ques,_=sentence_pad(ques_int,0)
        ques=torch.tensor(ques).to(self.device)
        return ques
    def forward(self, question):
        ques=self.ques2id(question)
        ques_embed=self.embedding(ques)
        # print(ques_embed.device)
        # print(ques_embed.shape)
        # h0,c0=self.encoder.init_hidden()
        # hn,cn=self.encoder(ques_embed,np.array(ques_len),h0,c0)
        # return hn,cn
        return ques_embed

class BertEncoder(nn.Module):
    def __init__(self,hidden_size,device):
        super(BertEncoder, self).__init__()

        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert=BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size=hidden_size
        self.device=device

    def ques_token2id(self,ques):
        token_data = []
        for row in ques:
            tokenized = self.tokenizer.tokenize(row)
            tokenized.insert(0, '[CLS]')
            indexed = self.tokenizer.convert_tokens_to_ids(tokenized)
            token_data.append(indexed)
        return token_data

    def concat(self,layer1,layer2):
        layer1=torch.mean(layer1,dim=1)
        layer2=torch.mean(layer2,dim=1)
        layer=(layer1+layer2)/2
        return layer

    def forward(self, ques):
        token_data=self.ques_token2id(ques)
        ques_id,ques_mask=torch.tensor(padding(token_data)).to(self.device)
        encoder_layers,_=self.bert(input_ids=ques_id,attention_mask=ques_mask)
        layer1,layer2=encoder_layers[-1],encoder_layers[-2]
        layer=self.concat(layer1,layer2)
        return layer
