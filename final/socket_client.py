
import cv2
import json
import numpy as np
import os
import socket
import struct
import zipfile

def train(fileName):
    # input file
    with open(fileName, 'r') as f:
        print("Loading descriptors...")
        input_file = json.load(f)
        x = np.array(input_file["train_x"], dtype = np.float32)
        y = np.array(input_file["train_y"], dtype = np.int32)

    # 訓練模型
    print("Training")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)    # Set SVM type
    svm.setKernel(cv2.ml.SVM_LINEAR) # Set SVM Kernel
    svm.setC(0.1)                    # Set parameter C
    svm.setGamma(1.0)                # Set parameter Gamma
    output_model_path = input_file["model_name"] + ".xml"
    svm.trainAuto(x, cv2.ml.ROW_SAMPLE, y)
    svm.save(output_model_path)
    return input_file["model_name"] + ".xml"

host = '163.18.49.33'
port = 8001
fmt = '128si'
buffer_size = 8096
 
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn.connect((host, port))
while 1:
    headsize = struct.calcsize(fmt)
    head = conn.recv(headsize)
    print("Work start")
    filename = struct.unpack(fmt, head)[0].decode().rstrip('\0')
    filesize = struct.unpack(fmt, head)[1]
    print("filename:" + filename + "\nfilesize:" + str(filesize))
    recved_size = 0
    fd = open(filename, 'wb')
    count = 0
    while True:
        data = conn.recv(buffer_size)
        recved_size = recved_size + len(data)
        fd.write(data)
        if recved_size == filesize:
            break
    fd.close()
    print("Recieve complete")
    model_name = train(filename)
    print("Train complete")

    filesize = os.path.getsize(model_name)
    print("model_name:" + model_name + "\nfilesize:" + str(filesize))
    head = struct.pack(fmt, model_name.encode(), filesize)
    conn.sendall(head)
    restSize = filesize
    fd = open(model_name,'rb')
    count = 0
    while restSize >= buffer_size:
        data = fd.read(buffer_size)
        conn.sendall(data)
        restSize = restSize - buffer_size
        count = count + 1
    data = fd.read(restSize)
    conn.sendall(data)
    fd.close()
    # 刪除模型
    os.remove(model_name)
    print("Work complete")
