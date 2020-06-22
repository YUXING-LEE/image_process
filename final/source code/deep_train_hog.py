# coding=utf8
import cv2
import json
import numpy as np
import os
import time
import multiprocessing as mp

#------------------------ Define ------------------------#
total_time = time.time()
# input path
input_folder = "./assets/origin/"
category_array = os.listdir(input_folder)
# load deep training setting
with open("setting.json", 'r') as f:
    deep_layer = json.load(f)

def findLayer(data, parentLabel):

    global pool

    train_x = []
    train_y = []
    for child in data["children"]:
        label = child["label"]
        for key in child["value"]:
            if key in descriptors_list.keys():
                buffer_x = []
                buffer_x += descriptors_list[key]
                buffer_y = [label] * len(buffer_x)
                train_x += buffer_x
                train_y += buffer_y

        if len(child["children"]) > 0:
            findLayer(child, parentLabel + "-" + label)

    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)

    print("train_x: ", train_x.shape)
    print("train_y: ", train_y.shape)
    pool.apply_async(export, (train_x, train_y, "test_" + parentLabel, ))

def export(train_x, train_y, fileName):
    with open("./work/" + fileName + ".json", 'w+') as f:
        request = {
            "model_name": fileName,
            "train_x": train_x.tolist(),
            "train_y": train_y.tolist()
        }
        json.dump(request, f)

    time.sleep(1)
    with open("./work.txt", 'a') as f:
        f.write(fileName + "\n")

    print(fileName + " work create! \n")

def addTrainSet(src, winSize):
    img = cv2.resize(src, winSize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = HOGDescriptor.compute(gray)
    descriptors = descriptors.ravel().tolist()
    return descriptors

if __name__ == '__main__':
    descriptors_list = {}
    winSize = (64, 64)
    HOGParams = {
        "_winSize": (64, 64),
        "_blockSize": (16, 16),
        "_blockStride": (8, 8),
        "_cellSize":  (8, 8), 
        "_nbins": 9
    }
    HOGDescriptor = cv2.HOGDescriptor(**HOGParams)
    for category in category_array:
        if ".xml" in category:
            continue

        path = input_folder + category + "/"
        for filename in os.listdir(path):
            src = cv2.imread(path + filename)
            svm_x = addTrainSet(src, winSize)
            try:
                descriptors_list[category].append(svm_x)
            except KeyError:
                descriptors_list[category] = [svm_x]

        print("Category: ", category, " has loaded!")

    pool = mp.Pool(processes=4)
    findLayer(deep_layer, "root")
    pool.close()
    pool.join()
    while 1:
        if len(os.listdir("./result/")) == 4:
            break
        else:
            time.sleep(1)
    print ("total time:", (time.time() - total_time))