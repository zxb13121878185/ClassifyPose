#!/usr/bin/env python
# -*- coding=utf-8 -*-
 
import socket
import sys
import numpy as np
import urllib
import cv2 as cv
import threading
import time

print('this is client')
 
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 260)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 200)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname() # 获取本地主机名
port = 6666  
s.connect((host, port))     #连接服务端
while True:
    if cv.waitKey(10) & 0xFF == ord('q'):
        sys.exit(1)  
        # get a frame
    ret, frame = cap.read()
        # '.jpg'表示把当前图片img按照jpg格式编码，按照不同格式编码的结果不一样
    img_encode = cv.imencode('.jpg', frame)[1] 
    data_encode = np.array(img_encode)
    str_encode = data_encode.tobytes()
    encode_len = str(len(str_encode))
      #  print('img size : %s'%encode_len)
    try:
        s.send(str_encode)#发送图片的encode码
    except Exception as e:
        print(e)
        sys.exit(1)
    time.sleep(0.1)
s.close()

 
