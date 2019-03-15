import json
import os
import sys
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
