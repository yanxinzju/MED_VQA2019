import csv
import os
import re
from torch.utils.data import Dataset
del_p='[,?]'
class med_data(Dataset):
    def __init__(self,csv_path,mode):
        root_path=os.path.join('../data',mode,'QA')
        csv_file=os.path.join(root_path,csv_path)
        self.mode=mode
        self.csv=csv.reader(open(csv_file))
        data=[]
        for row in self.csv:
            line=''.join(row).split('|')
            line[1]=re.sub(del_p,'',line[1])
            if self.mode=='train' or self.mode=='val':
                data.append((line[0],line[1],line[2]))
            else:
                data.append((line[0],line[1]))
        self.datas=data
    def __getitem__(self, index):
        if self.mode=='train' or self.mode=='val':
            image_id,question,answer=self.datas[index]
            data={'image_name':image_id,'question':question,'answer':answer}
        else:
            image_id,question=self.datas[index]
            data={'image_name':image_id,'question':question}
        return data
    def __len__(self):
        return len(self.datas)



