import cv2
import numpy as np
import threading
import time

def OPEN(image, kernel_size):
    morph_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    result = cv2.morphologyEx(image, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    return result

def CLOSE(image, kernel_size):
    morph_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    return result

def start(image, kernel_size_list, data_size):
    print("use avx: ", cv2.useOptimized())
    print("single thread")
    image_copy = image.copy()
    for kernel_size in kernel_size_list:
        start_time = time.time()
        for _ in range(data_size):
            image_copy = OPEN(image_copy, kernel_size)
        end_time = time.time() - start_time
        print(str(kernel_size) + "x" + str(kernel_size), "OPEN cost time:", end_time)
        cv2.imwrite("result/open_" + str(kernel_size) + ".png", image_copy)

    print("\n")
    image_copy = image.copy()
    for kernel_size in kernel_size_list:
        start_time = time.time()
        for _ in range(data_size):
            image_copy = CLOSE(image_copy, kernel_size)
        end_time = time.time() - start_time
        print(str(kernel_size) + "x" + str(kernel_size), "CLOSE cost time:", end_time)
        cv2.imwrite("result/close_" + str(kernel_size) + ".png", image_copy)

    print("\n")
    print("multi thread")
    for kernel_size in kernel_size_list:
        for thread_count in range(1, 6):
            start_time = time.time()
            work_count = 0
            while work_count <= data_size:
                threads = []
                for i in range(thread_count):
                    threads.append(threading.Thread(target = OPEN, args = (image, kernel_size, )))
                    threads[i].start()
                for i in range(thread_count):
                    threads[i].join()
                    work_count += 1
            end_time = time.time() - start_time
            print(str(kernel_size) + "x" + str(kernel_size), "thread:", thread_count, "OPEN cost time:", end_time)
    print("\n")

    for kernel_size in kernel_size_list:
        for thread_count in range(1, 6):
            start_time = time.time()
            work_count = 0
            while work_count <= data_size:
                threads = []
                for i in range(thread_count):
                    threads.append(threading.Thread(target = CLOSE, args = (image, kernel_size, )))
                    threads[i].start()
                for i in range(thread_count):
                    threads[i].join()
                    work_count += 1
            end_time = time.time() - start_time
            print(str(kernel_size) + "x" + str(kernel_size), "thread:", thread_count, "CLOSE cost time:", end_time)
    print("\n")

def main():
    # load image
    origin = cv2.imread("lena.jpg")
    # convert to gray
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result/gray.png", gray)
    # binary
    thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("result/threshold.png", thres)
    # set kernel size
    kernel_size_list = [3, 5, 7, 9]
    # enable avx
    cv2.setUseOptimized(True)
    start(thres, kernel_size_list, 100)
    # disable avx
    cv2.setUseOptimized(False)
    start(thres, kernel_size_list, 100)

if __name__ == '__main__':
    main()