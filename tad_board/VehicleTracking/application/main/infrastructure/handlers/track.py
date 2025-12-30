# limit the number of cpus used by high performance libraries

import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
# sys.path.insert(0, './yolov5')
lib_path = os.path.abspath(os.path.join('infrastructure', 'yolov5'))
sys.path.append(lib_path)

from tad_board.VehicleTracking.application.main.infrastructure.yolov5.models.common import DetectMultiBackend
from tad_board.VehicleTracking.application.main.infrastructure.yolov5.utils.datasets import LoadImages, LoadStreams
from tad_board.VehicleTracking.application.main.infrastructure.yolov5.utils.general import LOGGER, check_img_size, increment_path, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path
from tad_board.VehicleTracking.application.main.infrastructure.yolov5.utils.torch_utils import select_device, time_sync
from tad_board.VehicleTracking.application.main.infrastructure.yolov5.utils.plots import Annotator, colors

from tad_board.VehicleTracking.application.main.infrastructure.deep_sort_pytorch.utils.parser import get_config
from tad_board.VehicleTracking.application.main.infrastructure.deep_sort_pytorch.deep_sort import DeepSort
import argparse
import glob

from tad_board.VehicleTracking.application.main.util.common import  read_yml, extract_xywh_hog
from tad_board.VehicleTracking.application.main.util.OPT_config import OPT


from threading import Thread
from datetime import timedelta, datetime

import platform
import shutil
from pathlib import Path
import cv2
import copy
import numpy as np
# numpy>=1.24 removed aliases like np.float/np.int; keep compatibility for deep_sort code.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
import time
# import dlib

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tad_board.VehicleTracking.application.main.infrastructure.yolov5.utils.augmentations import letterbox

class Yolo5Tracker:
    def __init__(self, config_path:str) -> None:

        config = read_yml(config_path)
        
        self.opt = OPT(config=config)
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        self.device = select_device(self.opt.device)
        self.mode_device = select_device(self.device)
        
        self.mode = DetectMultiBackend(os.path.abspath(self.opt.yolo_weights), device= self.device, dnn=self.opt.dnn)

        # Older checkpoints may miss Upsample.recompute_scale_factor; ensure it exists to avoid AttributeError.
        if hasattr(self.mode, "model"):
            for m in self.mode.model.modules():
                if isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
                    m.recompute_scale_factor = None

        self.cfg = get_config()
        # Resolve deep_sort config relative to the provided config file location
        cfg_path = self.opt.config_deepsort
        if not os.path.isabs(cfg_path):
            cfg_dir = os.path.dirname(config_path)
            candidate = os.path.abspath(os.path.join(cfg_dir, os.path.basename(cfg_path)))
            cfg_path = candidate if os.path.exists(candidate) else os.path.abspath(cfg_path)
        else:
            cfg_path = os.path.abspath(cfg_path)
        self.cfg.merge_from_file(cfg_path)
        # Resolve reid checkpoint robustly (handles both relative and absolute entries)
        reid_ckpt = self.cfg.DEEPSORT.REID_CKPT
        candidates = []
        candidates.append(os.path.abspath(reid_ckpt))
        base_dir = os.path.abspath(os.path.join(os.path.dirname(config_path), os.pardir, os.pardir))  # project root (tad_board)
        candidates.append(os.path.abspath(os.path.join(base_dir, reid_ckpt)))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(config_path), os.path.basename(reid_ckpt))))
        candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", os.path.basename(reid_ckpt))))
        reid_ckpt_resolved = next((c for c in candidates if os.path.exists(c)), candidates[0])
        print("DeepSort ckpt candidates:", candidates)
        print("DeepSort ckpt resolved:", reid_ckpt_resolved)

        self.deepsort = DeepSort(reid_ckpt_resolved,
                            max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                            max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        self.previous_frame = -1
        self.current_frame = -1
        self.vehicles_count = 0
        self.IDs_vehicles = []
        self.list_vehicles = set()   #LIST CONTAIN vehicles HAS APPEARED, IF THAT VEHICLE HAD BEEN UPLOADED TO DB, REMOVE THAT VEHICLE
        self.vehicle_infos = {} # id:{start in view, exit view, type }

        # 判断是否进行half操作
        self.pt = self.mode.pt
        self.half = False
        self.half &= self.device.type != 'cpu' 
        self.half &= self.pt and self.mode_device.type != 'cpu'  # half precision only suvehiclesorted by PyTorch on CUDA
        # print('half', self.half)
        # self.half = True
        if self.pt:
            self.mode.model.half() if self.half else self.mode.model.float()
        self.imgsz = check_img_size(self.opt.imgsz, s=self.mode.stride)  # check image size

        if self.pt and self.mode_device.type != 'cpu':
            self.mode(torch.zeros(1, 3, *(self.imgsz)).to(self.mode_device).type_as(next(self.mode.model.parameters())))  # warmup

    def detect(self, frame_image, frame_idx, return_labels=False):
        opt = self.opt
        out, save_vid, evaluate, half = \
            opt.output, opt.save_vid, opt.evaluate, opt.half
        # zone_drawer = ZoneDrawerHelper()

        # Initialize
        if not evaluate:
            if os.path.exists(out):
                pass
            else:
                os.makedirs(out)  # make new output folder

        # Load model
        device = self.mode_device
        stride, names, jit, onnx = self.mode.stride, self.mode.names, self.mode.jit, self.mode.onnx
        
        if save_vid:
            cv2.namedWindow("detect trace video")
        # for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        im0s = frame_image
        # Padded resize
        img = letterbox(im0s, 640, 32, True)[0]  # 取自datasets.py 中的 LoadImages类  __next__函数

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.mode(img, augment=opt.augment, visualize=False)
        # Avehiclesly NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        # 返回的值 frame_id track_id (1,4)方框坐标
        trace_data = []
        detected_labels = set()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s.copy()

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # # pass detections to deepsort, only objects in used zone
                xywhs = np.asarray(xywhs.cpu())
                confs = np.asarray(confs.cpu())
                clss = np.asarray(clss.cpu())
                xywhs = torch.tensor(xywhs)

                confs = torch.tensor(confs)
                clss = torch.tensor(clss)

                outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                self.current_frame = {}
                self.current_frame['frame'] = frame_idx
                self.current_frame['n_vehicles_at_time'] = len(outputs)
                self.current_frame['IDs_vehicles'] = []

                if len(outputs) > 0:
                    self.current_frame['IDs_vehicles'] = list(outputs[:, 4])
                    # current_frame['bb_vehicles'] = list(outputs[:, :4])

                if (self.current_frame != -1) and (self.previous_frame != -1):
                    previous_IDs = self.previous_frame['IDs_vehicles']
                    current_IDs = self.current_frame['IDs_vehicles']

                    for ID in current_IDs:
                        # neu id khong co trong khung hinh truoc va chua tung xuat hien
                        if (ID not in previous_IDs) and (ID not in self.list_vehicles):
                            self.vehicle_infos[ID] = {}
                            self.vehicle_infos[ID]['in_time'] = datetime.now()
                            self.vehicle_infos[ID]['exit_time'] = datetime.max
                            self.vehicle_infos[ID]['type_vehicle'] = 'vehicle'
                            self.vehicle_infos[ID]['lane'] = 'lane'
                            self.vehicle_infos[ID]['temporarily_disappear'] = 0

                    # for ID in previous_IDs:
                    for ID in copy.deepcopy(self.list_vehicles):
                        if (ID not in current_IDs):
                            self.vehicle_infos[ID]['exit_time'] = datetime.now()
                            self.vehicle_infos[ID]['temporarily_disappear'] += 1
                            # 25 frame ~ 1 seconds
                            if (self.vehicle_infos[ID]['temporarily_disappear'] > 75) and \
                                    (self.vehicle_infos[ID]['exit_time'] - self.vehicle_infos[ID]['in_time']) > timedelta(
                                seconds=3):
                                self.list_vehicles.discard(ID)
                                # vehicle_infos.pop(ID)

                # Visualize deepsort outputs
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        if 0 <= c < len(names):
                            detected_labels.add(names[c])
                        # label = f'{id} {names[c]} {conf:.2f}'
                        label = f'{names[c]}- id {id}'

                        # 此处label即为左上角ID
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        frameidx_id_box = np.concatenate([np.array([frame_idx, id]), bboxes])
                        # 输出检测框和标识号
                        print("id", id)
                        print("标识框", bboxes)
                        # print("outputs", outputs)
                        self.vehicle_infos[id]['type_vehicle'] = names[c]
                        trace_data.append(frameidx_id_box)
                    self.vehicles_count, self.IDs_vehicles = self.current_frame['n_vehicles_at_time'], self.current_frame['IDs_vehicles']

                    if not np.isnan(np.sum(self.IDs_vehicles)):
                        self.list_vehicles.update(list(self.IDs_vehicles))
            else:
                self.deepsort.increment_ages()

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_vid:
                # print(im0.shape)
                cv2.imshow("detect trace video", im0)  # 显示窗口的名字， 所要显示的图片
                # cv2.imwrite("/home/oseasy/GraduationProject/tad-IROS2019-combination/detect_trace/"+str(frame_idx)+".jpg", im0)
                cv2.waitKey(1)

        self.previous_frame = self.current_frame

        # print(self.vehicle_infos)
        # print(self.list_vehicles)
        if return_labels:
            return im0, sorted(detected_labels)
        return im0
        # cv2.destroyWindow("detect trace video")
