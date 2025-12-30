import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import os
import numpy as np
import yaml
import glob
import pickle as pkl
import random
import cv2
import copy
from PIL import Image
import torch
import argparse
from torch.utils import data
from config.config import *


from config.config import parse_args, visualize_config
import time
import pyrealsense2 as rs
# import torch_tensorrt
# 下面为目标检测，轨迹跟踪引入模块
#lib_path = os.path.abspath(os.path.join('VehicleTracking/application/main/infrastructure', 'yolov5'))  #添加到环境变量
#sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('VehicleTracking/application', 'main'))  #添加到环境变量
sys.path.append(lib_path)
from infrastructure.handlers.track import Yolo5Tracker


from threading import Thread

print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 初始化目标检测，轨迹跟踪
tracker = Yolo5Tracker(config_path="VehicleTracking/settings/config.yml")
current_directory = os.path.dirname(os.path.abspath(__file__))


def anomaly_detect(args, visualize=False):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    # 随机生成颜色
    tracker_colors = []
    for i in range(999):
        tracker_colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    # 起始阶段未检测到标识框
    start_not_detected = False  

    WIDTH = args.W
    HEIGHT = args.H
    if args.input_type == 0:
        FPS = 30
        capture = cv2.VideoCapture(0)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    current_frame_id = 0
    while True:
        current_frame_id += 1

        if args.input_type == 0:
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 进行检测

        # frame_image = frame_image[:, :, ::-1]
        print("-----------------------------第" + str(current_frame_id) + "帧:-----------------------------")
        result = tracker.detect(frame, current_frame_id)
        show_result = result[...,::-1]
        if visualize:
            cv2.imshow("accident anomaly probability", show_result)
            cv2.waitKey(1)

if __name__ == '__main__':
    args = parse_args()
    anomaly_detect(args, visualize=True)
