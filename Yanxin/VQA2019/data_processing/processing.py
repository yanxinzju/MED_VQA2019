import sys
sys.path.append('..')
from model.util import *
def gen_all_embedding():
    glove = json.load(open('../file/glove.json', encoding='UTF-8'))
    gen_embedding('Abnormality',glove)
    gen_embedding('Modality',glove)
    gen_embedding('Organ',glove)
    gen_embedding('Plane',glove)

def gen_all_answer_set():
    gen_answer_set('Modality')
    gen_answer_set('Abnormality')
    gen_answer_set('Organ')
    gen_answer_set('Plane')

if __name__ == '__main__':
    # gen_all_embedding()
    # gen_all_answer_set()
    answer_set1=load_answer_set('Modality')
    answer_set2=load_answer_set('Abnormality')
    answer_set3=load_answer_set('Organ')
    answer_set4=load_answer_set('Plane')
    # print(answer_set1)
    # print(answer_set2)
    # print(answer_set3)
    # print(answer_set4)
