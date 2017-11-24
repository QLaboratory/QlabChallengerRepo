# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('qt4agg')
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

# zhfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')


OFFICIAL_CLASSIFICATION_RAW_FILE_PATH = r"D:\QlabChallengerRepo\ai_challenger_scene\classification_statistics\scene_validation_dictionaries.json"
PREDICTION_CLASSIFICATION_FILE_PATH = r"D:\QlabChallengerRepo\ai_challenger_scene\classification_statistics\ResNet152_predict_validation.json"
SCENE_CLASSES_RAW_FILE_PATH = r"D:\QlabChallengerRepo\ai_challenger_scene\classification_statistics\scene_classes.json"

# OFFICIAL_CLASSIFICATION_RAW_FILE_PATH = r"C:\Users\Air\Desktop\classification_statistics\scene_validation_dictionaries.json"
# PREDICTION_CLASSIFICATION_FILE_PATH = r"C:\Users\Air\Desktop\ALL\ResNet152_predict_validation.json"
# SCENE_CLASSES_RAW_FILE_PATH = r"C:\Users\Air\Desktop\classification_statistics\scene_classes.json"

sceneClassesFile = open(SCENE_CLASSES_RAW_FILE_PATH, 'r', encoding='UTF-8')
sceneClassesJSON = json.load(sceneClassesFile)

officialClassificationFile = open(OFFICIAL_CLASSIFICATION_RAW_FILE_PATH, 'r', encoding='UTF-8')
officialClassificationDICT = json.load(officialClassificationFile)

predictionClassificationFile = open(PREDICTION_CLASSIFICATION_FILE_PATH, 'r', encoding='UTF-8')
predictionClassificationLIST = json.load(predictionClassificationFile)

# 存储场景分类错误单张影像的详细信息
ClassificationErrorDetailDictionary = []
# 存储场景分类错误统计信息
ClassificationErrorStatisticsMatrix = np.zeros((80, 84))

# 遍历预测的JSON文件
for i in range(len(predictionClassificationLIST)):
    # 获取当前场景的正确分类ID
    image_id_temp = predictionClassificationLIST[i]['image_id']
    officail_label_id_temp = officialClassificationDICT[image_id_temp]
    ClassificationErrorStatisticsMatrix[int(officail_label_id_temp), 1] += 1
    # 获取当前场景的预测分类ID
    predict_label_id_temp = predictionClassificationLIST[i]['label_id']
    prediction_bool_temp = False
    predict_label_id_sort = np.asarray(predict_label_id_temp).argsort()[-3:][::-1].tolist()

    if int(officail_label_id_temp) in predict_label_id_sort:
        prediction_bool_temp = True

    classification_error_temp = {}
    if prediction_bool_temp is False:
        # 统计分类错误场景信息
        ClassificationErrorStatisticsMatrix[int(officail_label_id_temp), 2] += 1
        # 格式化输出分类错误场景信息
        classification_error_temp['image_id'] = image_id_temp
        classification_error_temp['label_id'] = officail_label_id_temp
        classification_error_temp['label_id_name'] = sceneClassesJSON[officail_label_id_temp]

        # 太长，暂不输出预测概率
        # classification_error_temp['label_id_predict'] = officail_label_id_temp

        label_id_predict_name_temp = []
        for k in predict_label_id_sort:
            label_id_predict_name_temp.append(sceneClassesJSON[str(k)])
            ClassificationErrorStatisticsMatrix[int(officail_label_id_temp), k+4] += 1

        classification_error_temp['label_id_predict_name'] = label_id_predict_name_temp
        # print(classification_error_temp)
        ClassificationErrorDetailDictionary.append(classification_error_temp)


# 存储场景分类错误统计输出信息
ClassificationErrorStatisticDictionary = []
for i in range(80):
    ClassificationErrorStatisticsMatrix[i, 0] = i
    ClassificationErrorStatisticsMatrix[i, 3] = np.true_divide(ClassificationErrorStatisticsMatrix[i, 2], ClassificationErrorStatisticsMatrix[i, 1]) * 100

    classification_error_statistic_temp = {}
    classification_error_statistic_temp['label_id'] = i
    classification_error_statistic_temp['label_id_name'] = sceneClassesJSON[str(i)]
    classification_error_statistic_temp['total_number'] = ClassificationErrorStatisticsMatrix[i, 1]
    classification_error_statistic_temp['classification_error_number'] = ClassificationErrorStatisticsMatrix[i, 2]
    classification_error_statistic_temp['classification_error_percentage'] = ClassificationErrorStatisticsMatrix[i, 3]

    classification_error_statistic_temp_list = ClassificationErrorStatisticsMatrix[i, 4:len(ClassificationErrorStatisticsMatrix[0])]
    classification_error_statistic_temp_list_sort = classification_error_statistic_temp_list.argsort()[-3:][::-1].tolist()
    classification_error_statistic_temp['classification_error_statistic_label_id'] = classification_error_statistic_temp_list_sort
    classification_error_statistic_name_temp = []
    for k in classification_error_statistic_temp_list_sort:
        classification_error_statistic_name_temp.append(sceneClassesJSON[str(k)])
    classification_error_statistic_temp['classification_error_statistic_label_name'] = classification_error_statistic_name_temp
    ClassificationErrorStatisticDictionary.append(classification_error_statistic_temp)
    # print(classification_error_statistic_temp)

# 输出场景分类错误，单张图片的详细信息
(filepath, tempfilename) = os.path.split(PREDICTION_CLASSIFICATION_FILE_PATH)
(shotname, extension) = os.path.splitext(tempfilename)
CLASSIFICATION_ERROR_DETAIL_STATISTICS_FILE_PATH = shotname + "_classification_error_detail_statistics" + ".json"
CLASSIFICATION_ERROR_STATISTICS_FILE = open(CLASSIFICATION_ERROR_DETAIL_STATISTICS_FILE_PATH, "w", encoding='UTF-8')

# print(len(ClassificationErrorDetailDictionary))
json.dump(ClassificationErrorDetailDictionary, CLASSIFICATION_ERROR_STATISTICS_FILE, indent=2, ensure_ascii=False)

# 输出场景分类错误，每个场景的统计信息
CLASSIFICATION_ERROR_TOTAL_STATISTICS_FILE_PATH = shotname + "_classification_error_total_statistics" + ".json"
CLASSIFICATION_ERROR_TOTAL_STATISTICS_FILE = open(CLASSIFICATION_ERROR_TOTAL_STATISTICS_FILE_PATH, "w", encoding='UTF-8')
json.dump(ClassificationErrorStatisticDictionary, CLASSIFICATION_ERROR_TOTAL_STATISTICS_FILE, indent=2, ensure_ascii=False)

# 输出场景分类错误统计矩阵信息
CLASSIFICATION_ERROR_STATISTICS_MATRIX_FILE_PATH = shotname + "_classification_error_statistics" + ".txt"
# print(ClassificationErrorStatisticsMatrix)
# np.savetxt(CLASSIFICATION_ERROR_STATISTICS_MATRIX_FILE_PATH, ClassificationErrorStatisticsMatrix, fmt='%d', delimiter='\t', newline='\r\n')


ClassificationErrorStatisticsMatrixSort = ClassificationErrorStatisticsMatrix[ClassificationErrorStatisticsMatrix[:,3].argsort()]
np.savetxt(CLASSIFICATION_ERROR_STATISTICS_MATRIX_FILE_PATH, ClassificationErrorStatisticsMatrixSort, fmt='%d', delimiter='\t', newline='\r\n')

alphabetX = []
alphabetY = []
count = 0
for i in ClassificationErrorStatisticsMatrixSort:
    alphabetX.append(str(count))
    alphabetY.append(sceneClassesJSON[str(int(i[0]))])
    count += 1


def ConfusionMatrixPng(cm, ):

    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i)
        # print(a)
        for j in i:
            tmp_arr.append(float(j) / (float(a) + 1.1e-5))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.gray_r,
                    interpolation='nearest')
    height = len(cm)
    width = len(cm[0])
    cb = fig.colorbar(res)
    # locs, labels = plt.xticks(range(width), alphabet[:width])
    # for t in labels:
    #     t.set_rotation(90)
    # plt.xticks('orientation', 'vertical')
    # locs, labels = xticks([1,2,3,4], ['Frogs', 'Hogs', 'Bogs', 'Slogs'])
    # setp(alphabet, 'rotation', 'vertical')
    plt.xticks(range(width), alphabetX[:width])
    plt.yticks(range(height), alphabetY[:height])
    plt.savefig('confusion_matrix.png', format='png')
    plt.show()


ConfusionMatrixPng(ClassificationErrorStatisticsMatrix[:, 4:len(ClassificationErrorStatisticsMatrixSort[0])])
