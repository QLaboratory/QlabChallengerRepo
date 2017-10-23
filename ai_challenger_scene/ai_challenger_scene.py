import os
import pickle
import numpy as np

from PIL import Image


DATA_URL_SCENE_TRAIN = "ai_challenger_scene_train_20170904"
DATA_URL_SCENE_VALIDATION = "ai_challenger_scene_validation_20170904"
DATA_URL_SCENE_TRAIN = "ai_challenger_scene_test_20170904"


__all__ = ['SceneTrain']

def read_scene(filenames, scene_classnum):
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


class SceneBase:
    def __init__(self, train_or_validation_or_test, shuffle=True, dir=None, scene_classnum=80):
        assert train_or_validation_or_test in ['train', 'validation', 'test']
        assert scene_classnum == 80
        self.scene_classnum = scene_classnum
        if train_or_validation_or_test == 'train':
            self.fs = os.path.join('..', DATA_URL_SCENE_TRAIN)
        elif train_or_validation_or_test == 'validation':
            self.fs = os.path.join('..', DATA_URL_SCENE_VALIDATION)
        else:
            self.fs = os.path.join('..', DATA_URL_SCENE_TRAIN)
        self.train_or_validation_or_test = train_or_validation_or_test
        self.data = read_scene(self.fs, scene_classnum)
        self.dir = dir
        self.shuffle = shuffle

    def size(self):
        if self.train_or_validation_or_test == 'train':
            data_size = 53879
        elif self.train_or_validation_or_test == 'validation':
            data_size = 7120
        else:
            data_size = 7040
        return data_size
    
