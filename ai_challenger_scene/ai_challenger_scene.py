import os
import pickle
import numpy as np
import six

from PIL import Image


DATA_URL_SCENE_TRAIN = "ai_challenger_scene_train_20170904"
DATA_URL_SCENE_VALIDATION = "ai_challenger_scene_validation_20170904"
DATA_URL_SCENE_TEST = "ai_challenger_scene_test_20170904"


__all__ = ['SceneTrain']


def read_scene(filenames, scene_classnum, train_or_validation_or_test):
    assert scene_classnum == 80
    ret = []
    fo = open(filenames, 'rb')
    if six.PY3:
        dic = pickle.load(fo, encoding='bytes')
    else:
        dic = pickle.load(fo)
    data = dic[b'data']
    label = dic[b'labels']
    if train_or_validation_or_test == 'train':
        img_num = 53879
    elif train_or_validation_or_test == 'validation':
        img_num = 7120
    else:
        img_num = 7040
    fo.close()
    for k in range(img_num):
        img = data[k].reshape(3, 32, 32)
        img = np.transpose(img, [1, 2, 0])
        new_im = Image.fromarray(img)
        new_im.show()
        ret.append([img, label[k]])
    return ret


class SceneBase:
    def __init__(self, train_or_validation_or_test, shuffle=True, directory=None, scene_classnum=80):
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
        self.data = read_scene(self.fs, scene_classnum, train_or_validation_or_test)
        self.directory = directory
        self.shuffle = shuffle

    def size(self):
        if self.train_or_validation_or_test == 'train':
            data_size = 53879
        elif self.train_or_validation_or_test == 'validation':
            data_size = 7120
        else:
            data_size = 7040
        return data_size

    def get_data(self):
        idxs = np.arange(len(self.data))
        for k in idxs:
            yield self.data[k]

    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and validation and test) images of size 32x32x3
        """
        all_imgs = [x[0] for x in read_cifar(fnames, self.cifar_classnum)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean