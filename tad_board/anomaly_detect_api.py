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


import time
import pyrealsense2 as rs
# import torch_tensorrt
# 下面为目标检测，轨迹跟踪引入模块
#lib_path = os.path.abspath(os.path.join('VehicleTracking/application/main/infrastructure', 'yolov5'))  #添加到环境变量
#sys.path.append(lib_path)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "VehicleTracking", "application", "main"))  #添加到环境变量
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if MAIN_DIR not in sys.path:
    sys.path.append(MAIN_DIR)

try:
    from tad_board.VehicleTracking.application.main.infrastructure.handlers.track import Yolo5Tracker
except ImportError:
    from VehicleTracking.application.main.infrastructure.handlers.track import Yolo5Tracker

print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 初始化目标检测，轨迹跟踪
CONFIG_PATH = os.path.join(CURRENT_DIR, "VehicleTracking", "settings", "config.yml")
tracker = Yolo5Tracker(config_path=CONFIG_PATH)
current_directory = os.path.dirname(os.path.abspath(__file__))
global current_frame_id
current_frame_id = 0

def anomaly_detect(frame_image, visualize=False):
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # 随机生成颜色
    tracker_colors = []
    for i in range(999):
        tracker_colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    # if args.input_type == 0:
    #     FPS = 30
    #     capture = cv2.VideoCapture(0)
    #     ref, frame = capture.read()
    #     if not ref:
    #         raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    global current_frame_id
    current_frame_id += 1

    # if args.input_type == 0:
    #     ref, frame = capture.read()
    #     if not ref:
    #         break
    #     # 格式转变，BGRtoRGB
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     # 进行检测

    print("-----------------------------第" + str(current_frame_id) + "帧:-----------------------------")
    result_img, detected_labels = tracker.detect(
        frame_image,
        current_frame_id,
        return_labels=True,
    )
    show_result = result_img[..., ::-1]
    # if visualize:
    #     cv2.imshow("car detected", show_result)
    #     cv2.waitKey(1)
    return show_result, detected_labels
