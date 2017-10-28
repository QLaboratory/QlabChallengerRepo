import gc
import json
import numpy as np


TRUE_LABLES_PATH = r"C:\Users\Air\Desktop\metrics\scene_train_annotations_20170904.json"
PREDICT_LABLES_PATH = r"C:\Users\Air\Desktop\metrics\test.json"

#load lables files
true_lables_file = open(TRUE_LABLES_PATH, 'r')
loaded_true_lables_file = json.load(true_lables_file)

predict_lables_file = open(PREDICT_LABLES_PATH, 'r')
loaded_predict_lables_file = json.load(predict_lables_file)

true_lables_dic = {}
for i in range(len(loaded_true_lables_file)):
    true_lables_dic[loaded_true_lables_file[i]['image_id']] = loaded_true_lables_file[i]['label_id']

predict_lables_dic = {}
for i in range(len(loaded_predict_lables_file)):
    predict_lables_dic[loaded_predict_lables_file[i]['image_id']] = loaded_predict_lables_file[i]['label_id']

#start caculate top_k accuracy
true_num = 0

for key in predict_lables_dic:
    true_lable = int(true_lables_dic[key])
    predict_array = np.array(predict_lables_dic[key])

    flag = False
    for i in predict_array:
        if (i==true_lable):
            flag = True
    if flag:
        true_num+=1

print(true_num/(len(loaded_predict_lables_file)))

gc.collect()

