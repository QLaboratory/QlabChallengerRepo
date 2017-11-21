# -*- coding: utf-8 -*-
import os
import json
import numpy as np

OFFICIAL_CLASSIFICATION_RAW_FILE_PATH = r"C:\Users\Air\Desktop\classification_statistics\scene_validation_dictionaries.json"
PREDICTION_CLASSIFICATION_FILE_PATH = r"C:\Users\Air\Desktop\ALL\ResNet152_predict_validation.json"
SCENE_CLASSES_RAW_FILE_PATH = r"C:\Users\Air\Desktop\classification_statistics\scene_classes.json"

sceneClassesFile = open(SCENE_CLASSES_RAW_FILE_PATH, 'r', encoding='UTF-8')
sceneClassesJSON = json.load(sceneClassesFile)

officialClassificationFile = open(OFFICIAL_CLASSIFICATION_RAW_FILE_PATH, 'r', encoding='UTF-8')
officialClassificationDICT = json.load(officialClassificationFile)

predictionClassificationFile = open(PREDICTION_CLASSIFICATION_FILE_PATH, 'r', encoding='UTF-8')
predictionClassificationLIST = json.load(predictionClassificationFile)

ClassificationErrorDictionary = []
for i in range(len(predictionClassificationLIST)):
    image_id_temp = predictionClassificationLIST[i]['image_id']
    officail_label_id_temp = officialClassificationDICT[image_id_temp]
    predict_label_id_temp = predictionClassificationLIST[i]['label_id']
    prediction_bool_temp = False
    predict_label_id_sort = np.asarray(predict_label_id_temp).argsort()[-3:][::-1].tolist()

    if int(officail_label_id_temp) in predict_label_id_sort:
        prediction_bool_temp = True

    classification_error_temp = {}
    if prediction_bool_temp is False:
        classification_error_temp['image_id'] = image_id_temp
        classification_error_temp['label_id'] = officail_label_id_temp
        classification_error_temp['label_id_name'] = sceneClassesJSON[officail_label_id_temp]

        # 太长，暂不输出预测概率
        # classification_error_temp['label_id_predict'] = officail_label_id_temp

        label_id_predict_name_temp = []
        for k in predict_label_id_sort:
            label_id_predict_name_temp.append(sceneClassesJSON[str(k)])

        classification_error_temp['label_id_predict_name'] = label_id_predict_name_temp
        print(classification_error_temp)
        ClassificationErrorDictionary.append(classification_error_temp)

(filepath, tempfilename) = os.path.split(PREDICTION_CLASSIFICATION_FILE_PATH)
(shotname, extension) = os.path.splitext(tempfilename)
CLASSIFICATION_ERROR_STATISTICS_FILE_PATH = PREDICTION_CLASSIFICATION_FILE_PATH + "_classification_error_statistics" + ".json"
CLASSIFICATION_ERROR_STATISTICS_FILE = open(CLASSIFICATION_ERROR_STATISTICS_FILE_PATH, "w", encoding='UTF-8')
json.dump(ClassificationErrorDictionary, CLASSIFICATION_ERROR_STATISTICS_FILE, indent=2, ensure_ascii=False)

print(len(ClassificationErrorDictionary))


        






