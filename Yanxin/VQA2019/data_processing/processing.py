import sys
sys.path.append('..')
from model.util import *
def gen_all_embedding():
    glove = json.load(open('../file/glove.json', encoding='UTF-8'))
    gen_embedding('Abnormality',glove)
    gen_embedding('Modality',glove)
    gen_embedding('Organ',glove)
    gen_embedding('Plane',glove)
if __name__ == '__main__':
    gen_all_embedding()