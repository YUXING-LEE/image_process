import os
import time
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
        fd = open("./result/" + filename, 'wb')
        while True:
            data = conn.recv(buffer_size)
            recved_size = recved_size + len(data)
            fd.write(data)
            if recved_size == filesize:
                break
        fd.close()
        client_list[processID]["working"] = False
        print("Recieve complete")

def listen():
    global server
    while 1:
        new_road = server.accept()
        client_IP = new_road[1][0]
        threads[client_IP] = threading.Thread(target = onNewClient, args = (new_road, ))
        threads[client_IP].start()
        time.sleep(1)

threads = {}
client_list = {}

fmt = '128si'
buffer_size = 8096

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("192.168.101.116", 3000))
server.listen(1)

listen_server = threading.Thread(target = listen)
listen_server.start()

task_log = ['']
work_path = "./work/"
while 1:
    with open("work.txt", 'r') as f:
        task_list = f.read()
        task_list = task_list.split("\n")
        for task in task_list:
            if task not in task_log or task == "":
                break

        if task != "":
            for clientID in dict.keys(client_list):
                client = client_list[clientID]
                if not client["working"]:
                    sock = client["connect"]
                    filename = task + ".json"
                    filesize = os.path.getsize(work_path + filename)
                    print("filename:" + filename + "\nfilesize:" + str(filesize))
                    head = struct.pack(fmt, filename.encode(), filesize)
                    sock.send(head)
                    restSize = filesize
                    fd = open(work_path + filename, 'rb')
                    while restSize >= buffer_size:
                        data = fd.read(buffer_size)
                        sock.send(data)
                        restSize = restSize - buffer_size
                    data = fd.read(restSize)
                    sock.send(data)
                    fd.close()
                    print("successfully sent")
                    client["working"] = True
                    task_log.append(task)
                    break
                else:
                    print("task busy")
    time.sleep(1)