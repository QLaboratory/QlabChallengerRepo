import os
import pickle
from PIL import Image
import numpy as np
import json

IMAGE_SIZE = 224
# DIR = "C:\\Users\\Air\\Desktop\\pickle_test\\dataset"
# ANNOTATION_FILE = "./train_annotations.json"
# ANNOTATION_DICT_FILE = "./train_dict_annotations.json"
DIR = "/home/yan/Desktop/QlabChallengerRepo/dataset/scene_train_images_20170904_direct_resize"
ANNOTATION_FILE = "./scene_train_annotations_20170904.json"
ANNOTATION_DICT_FILE = "./scene_train_annotations_dict_20170904.json"
OUTPUT_FILE = "ai_challenger_scene_train_direct_resize"


def ConvertImgToArray(filename):

    im = Image.open(DIR + "/" + filename)
    im_array = np.array(im)

    im_array_r = im_array[:, :, 0]
    im_array_g = im_array[:, :, 1]
    im_array_b = im_array[:, :, 2]

    im_array_r_dim1 = im_array_r.reshape(IMAGE_SIZE * IMAGE_SIZE)
    im_array_g_dim1 = im_array_g.reshape(IMAGE_SIZE * IMAGE_SIZE)
    im_array_b_dim1 = im_array_b.reshape(IMAGE_SIZE * IMAGE_SIZE)

    im_array_res = np.concatenate(
        (im_array_r_dim1, im_array_g_dim1, im_array_b_dim1), axis=0)

    return im_array_res


def GetJpgList(p):
    if p == "":
        return []
    # p = p.replace("/", "\\")
    if p[-1] != "/":
        p = p + "/"
    file_list = os.listdir(p)
    jpg_list = []
    for i in file_list:
        if os.path.isfile(p + i):
            name, suffix = os.path.splitext(p + i)
            if ('.jpg' == suffix):
                jpg_list.append(i)
    return jpg_list


if os.path.exists(DIR):
    files = GetJpgList(DIR)
else:
    print('folder can not find')

data_matrix = np.zeros((len(files), IMAGE_SIZE * IMAGE_SIZE * 3), dtype='uint8')

# Read annotation file
load_annotation_file = open(ANNOTATION_FILE, 'r')
load_annotation_dict_file = open(ANNOTATION_DICT_FILE,"w")
load_annotation = json.load(load_annotation_file)

annotation_dic = {}

for i in range(len(load_annotation)):
    annotation_dic[load_annotation[i]['image_id']] = load_annotation[i]['label_id']

json.dump(annotation_dic, load_annotation_dict_file)

row_number = 0
data_labels = []

for i in files:
    data_labels.append(annotation_dic[i])
    im_array_res = ConvertImgToArray(i)
    data_matrix[row_number, :] = im_array_res
    row_number += 1
    print(i)

final_res = {b'data': data_matrix, b'labels': data_labels}
# Write to the file
f = open(OUTPUT_FILE, 'wb')

pickle.dump(final_res, f,protocol=4)  # dump the object to a file
f.close()

del final_res


#Read back from the storage
with open(OUTPUT_FILE, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

print(dict)
