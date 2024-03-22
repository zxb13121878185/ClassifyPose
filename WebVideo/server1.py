#!/usr/bin/env python
# -*- coding=utf-8 -*-
 
import socket
import numpy as np
import urllib
import cv2
import threading
import time
import sys

print('this is server')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s = socket.socket()         # 创建 socket 对象
    host = socket.gethostname() # 获取本地主机名
    port = 6666                 # 设置端口
    s.bind((host, port))        # 绑定端口
    s.listen(10)
except socket.error as msg:
    print (msg)
    sys.exit(1)
print ('Waiting connection...') 
while True:
    c,addr = s.accept()     # 建立客户端连接
    if cv2.waitKey(10) & 0xFF == ord('q'):
        sys.exit(1)  
    try:
        receive_encode = c.recv(77777)#接收的字节数 最大值 2147483647 （31位的二进制）
        nparr = np.frombuffer(receive_encode, np.uint8)
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow("Server_Show", img_decode)
        time.sleep(0.1)
        print ('接受数据...') 
    except Exception as e:
        print(e)
        sys.exit(1)
c.close()                # 关闭连接
 

 