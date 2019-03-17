import json
import os
import sys
import csv
import numpy as np
sys.path.append('..')

def gen_name2id(mode):
    path=os.path.join('../data',mode,'images')
    i = 0
    name = []
    num = []
    for file in os.listdir(path):
        file = file.split('.')[0]
        name.append(file)
        num.append(i)
        print(file)
        i += 1
    dic = dict(map(lambda x, y: [x, y], name, num))
    print(i)
    with open('../file/'+mode+'_name2id.json', 'w') as json_file:
        json_file.write(json.dumps(dic))

def load_name2id(mode):
    path='../file/'+mode+'_name2id.json'
    with open(path,encoding='UTF-8') as json_file:
        data=json.load(json_file)
    return data

def extract_word_vocab(csv_file,mode):
    # print(csv_file)
    data = csv.reader(open(csv_file,'r',encoding='ISO-8859-1'))
    special_words = ['<PAD>','<UNK>','<SOS>', '<EOS>']
    set_words=[]
    for row in data:
        question=''.join(row).split('|')[1].split()
        # print(question)
        for word in question:
            set_words.append(word)
    set_words=list(set(set_words))
    # print(set_words)
    print(len(set_words))
    int2vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab2int = {word: idx for idx, word in int2vocab.items()}
    with open('../file/'+mode+'/int2vocab.json','w') as json_file:
        json_file.write(json.dumps(int2vocab))
    with open('../file/'+mode+'/vocab2int.json','w') as json_file:
        json_file.write(json.dumps(vocab2int))

def gen_embedding(mode,glove):
    int2vocab=json.load(open('../file/'+mode+'/int2vocab.json',encoding='UTF-8'))
    vocab_list=[]
    vec_list=[]
    j=0
    for i in range(len(int2vocab)):
        vocab=int2vocab[str(i)]
        try:
            num=list(glove[vocab])
            j+=1
        except KeyError:
            num = list(np.random.normal(scale=0.6, size=(300,)))
        vec_list.append(num)
        vocab_list.append(vocab)
    dic=dict(map(lambda x,y:[x,y],vocab_list,vec_list))
    print('{}/{}'.format(j,len(int2vocab)))
    with open('../file/'+mode+'/embedding.json','w') as json_file:
        json_file.write(json.dumps(dic))
