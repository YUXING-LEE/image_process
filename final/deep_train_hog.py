# coding=utf8
import cv2
import json
import numpy as np
import os
import time

#------------------------ Define ------------------------#
total_time = time.time()
# input path
input_folder = "./assets/安明路/origin/"
category_array = os.listdir(input_folder)
# load deep training setting
with open("安明路_deep.json", 'r') as f:
    deep_layer = json.load(f)

def findLayer(data, parentLabel):
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
    with open("test_" + parentLabel + ".json", 'w+') as f:
        request = {
            "model_name": "test_" + parentLabel,
            "train_x": train_x.tolist(),
            "train_y": train_y.tolist()
        }
        json.dump(request, f)
        task_list.append("test_" + parentLabel + ".json")
    print(parentLabel + " train done! \n")

def addTrainSet(src, winSize):
    img = cv2.resize(src, winSize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = HOGDescriptor.compute(gray)
    descriptors = descriptors.ravel().tolist()
    return descriptors

# start socket
import threading
import socket
import struct
def onNewClient(newClient):
    conn, (clientIP, processID) = newClient
    print(clientIP, "has logined")
    client_list[processID] = {
        "connect": conn,
        "working": False
    }
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
        client_list[processID]["working"] = False
        print("Recieve complete")

def sendTask():
    global task_list, client_list, close
    while 1:
        if len(task_list) > 0:
            for clientID in dict.keys(client_list):
                client = client_list[clientID]
                if not client["working"]:
                    sock = client["connect"]
                    filepath = task_list[0]
                    filename = os.path.split(filepath)[1]
                    filesize = os.path.getsize(filepath)
                    print("filename:" + filename + "\nfilesize:" + str(filesize))
                    head = struct.pack(fmt, filename.encode(), filesize)
                    sock.sendall(head)
                    restSize = filesize
                    fd = open(filepath, 'rb')
                    count = 0
                    while restSize >= buffer_size:
                        data = fd.read(buffer_size)
                        sock.sendall(data)
                        restSize = restSize - buffer_size
                        count = count + 1
                    data = fd.read(restSize)
                    sock.sendall(data)
                    fd.close()
                    print("successfully sent")
                    client["working"] = True
                    task_list.pop(0)
                    break
                else:
                    print("task busy")
        if close:
            break
        time.sleep(1)

def listen():
    global server
    while 1:
        print("waiting...")
        new_road = server.accept()
        client_IP = new_road[1][0]
        threads[client_IP] = threading.Thread(target = onNewClient, args = (new_road, ))
        threads[client_IP].start()
        time.sleep(1)

close = False
threads = {}
client_list = {}
task_list = []

fmt = '128si'
buffer_size = 8096

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("163.18.49.33", 8001))
server.listen(1)

listen_server = threading.Thread(target = listen)
listen_server.start()
task_server = threading.Thread(target = sendTask)
task_server.start()

# start training
while 1:
    start_flag = input("enter 1 for start\n")
    if start_flag == "1":
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

        findLayer(deep_layer, "root")
        print ("total time:", (time.time() - total_time))
        
    elif start_flag == 0:
        close = True
        break