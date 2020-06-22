# coding=utf8
import cv2
import json
import numpy as np
import os

# input path
input_folder = "./assets/安明路_test/origin/"
category_array = os.listdir(input_folder)
# import model
deep_model = {}
model_path = "./model/安明路_deep/"
model_name_list = os.listdir(model_path)
model_name_list.sort(key = lambda x: len(x.split("-")))
for model_name in model_name_list:
    model_split = model_name[:-4].split("-")[1:]
    if len(model_split) == 0:
        deep_model = {
            "model": cv2.ml.SVM_load(model_path + model_name)
        }
    else:
        root = deep_model
        for category in model_split:
            try:
                root = root[category]
            except KeyError:
                root[category] = {}
        root[category] = {
            "model": cv2.ml.SVM_load(model_path + model_name)
        }
# create hog
winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
HOGDescriptor = cv2.HOGDescriptor( _winSize = winSize,
                                   _blockSize = blockSize,
                                   _blockStride = blockStride,
                                   _cellSize = cellSize, 
                                   _nbins = nbins)

def addMotionTrainSet(src, x, y, w, h, ratio, video_width = 960, video_height = 540):

    img = cv2.resize(src, winSize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_gray = np.zeros(winSize, dtype=np.float32)
    norm_gray = cv2.normalize(gray, norm_gray, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    motion_params = norm_gray.ravel()
    
    y_parameter = y / video_height
    w_parameter = w / video_width
    h_parameter = h / video_height
    image_params = [y_parameter, w_parameter, h_parameter, ratio]
    use_ratio_parameter = np.append(image_params, [motion_params])
    parameter = motion_params

    return parameter, use_ratio_parameter

def addHogTrainSet(src, x, y, w, h, ratio, video_width = 960, video_height = 540):
    img = cv2.resize(src, winSize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = HOGDescriptor.compute(gray)
    descriptors = descriptors.ravel()

    y_parameter = y / video_height
    w_parameter = w / video_width
    h_parameter = h / video_height
    parameter = descriptors
    image_params = [y_parameter, w_parameter, h_parameter, ratio]
    use_ratio_parameter = np.append(image_params, [descriptors])

    return parameter, use_ratio_parameter

def predict(deepModel, feature):
    label = str(int(deepModel["model"].predict(feature)[1][0][0]))
    try:
        return predict(deepModel[label], feature)
    except KeyError:
        return label

result = {}
result_score = {}
keys = []
image_counter = 0
# 分類初始化
for category in category_array:
    target = category if category == "999" else str(int(category) // 10)
    if target not in result:
        result[target] = { "TP": 0, "FP": 0, "TN": 0, "FN": 0, }
        result_score[target] = { "accuracy": 0, "precision": 0, "recall": 0, "F1Score": 0 }
        keys.append(target)

# 開始分類
for category in category_array:
    if ".xml" in category:
        continue
    
    target = int(category) if category == "999" else int(category) // 10
    path = input_folder + category + "/"
    for filename in os.listdir(path):
        _, parameter = filename.split("-")
        x, y, w, h, ratio = parameter.split("_")
        ratio = float(ratio[:-4])
        src = cv2.imread(path + filename)

        svm_x, use_ratio_svm_x = addHogTrainSet(src, int(x), int(y), int(w), int(h), ratio)

        svm_x = np.array([svm_x], dtype=np.float32)

        predict_y = predict(deep_model, svm_x)

        predict_y = int(predict_y) if predict_y == "999" else int(predict_y) // 10

        if target == predict_y:
            result[str(target)]["TP"] += 1
        else:
            result[str(target)]["FN"] += 1
            result[str(predict_y)]["FP"] += 1
        for key in keys:
            if key != str(target) and key != str(predict_y):
                result[key]["TN"] += 1
        image_counter += 1

for key in keys:
    category = result[key]

    category["TP"] = float(category["TP"])
    category["TN"] = float(category["TN"])
    category["FP"] = float(category["FP"])
    category["FN"] = float(category["FN"])

    result_score[key]["accuracy"] = (category["TP"] + category["TN"]) / image_counter
    try:
        result_score[key]["precision"] = category["TP"] / (category["TP"] + category["FP"])
    except:
        result_score[key]["precision"] = 0

    try:
        result_score[key]["recall"] = category["TP"] / (category["TP"] + category["FN"])
    except:
        result_score[key]["recall"] = 0

    try:
        result_score[key]["F1Score"] = 2 * (result_score[key]["precision"] * result_score[key]["recall"]) / (result_score[key]["precision"] + result_score[key]["recall"])
    except:
        result_score[key]["F1Score"] = 0

model_name = "test"
with open("report/" + model_name + "_report.json", 'w+') as f:
    json.dump(result_score, f)