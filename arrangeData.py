import os
from PIL import Image
import numpy as np
import json
import shutil

INPUT_DIR = "/home/yan/Desktop/QlabChallengerRepo/dataset_224/scene_train_cotent_resize/scene_train_images_20170904_content_resize/"
ANNOTATION_FILE = "./scene_train_annotations_20170904.json"
OUTPUT_DIR = "/home/yan/Desktop/QlabChallengerRepo/dataset_224/data/train/"

def GetJpgList(p):
    if p == "":
        return []
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

#test_path = '/home/yan/Desktop/test/'
#print(GetJpgList(test_path))

# Read annotation file
load_annotation_file = open(ANNOTATION_FILE, 'r')
load_annotation = json.load(load_annotation_file)
annotation_dic = {}
for i in range(len(load_annotation)):
    annotation_dic[load_annotation[i]['image_id']] = load_annotation[i]['label_id']

#print(annotation_dic['c8e5d25c472956be94f7ed433b66d1d1caa49bfb.jpg'])

#check origin file
if os.path.exists(INPUT_DIR):
    images = GetJpgList(INPUT_DIR)
else:
    print('folder can not find')

#check
if len(images)!=len(annotation_dic):
    print('Wrong Image Folder or Wrong Annotation File')
    
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for image in images:
    image_label = annotation_dic[image]
    image_path  = OUTPUT_DIR+str(image_label)
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    print(image_path)
    shutil.copy(INPUT_DIR+image,image_path+'/'+image)
    