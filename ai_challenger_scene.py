import pickle
import numpy as np
from PIL import Image

def read_cifar(filenames):
    ret = []
    fo = open(filenames, 'rb')
    dic = pickle.load(fo, encoding='bytes')
    data = dic[b'data']
    label = dic[b'labels']
    
    IMG_NUM = 53879
    fo.close()
    for k in range(IMG_NUM):
        img = data[k].reshape(3, 32, 32)
        img = np.transpose(img, [1, 2, 0])
        new_im = Image.fromarray(img)
        new_im.show()
        ret.append([img, label[k]])
    return ret

reader = read_cifar(r"d:\pickle_test\ai_challenger_scene_train_20170904")


class SceneBase():
    def __init__(self):
        
