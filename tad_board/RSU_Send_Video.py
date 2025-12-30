# -*- coding: UTF-8 -*-
import socket
import time
import sys
import cv2
import numpy as np
import struct
import threading
import os
import csv
import json

# Ensure local imports work whether run as package or script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# import pyrealsense2 as rs
try:
    from tad_board.anomaly_detect_api import anomaly_detect
except ImportError:
    from anomaly_detect_api import anomaly_detect

IP_ADDRESS = '10.247.51.238'
# IP_ADDRESS = "127.0.0.1"
PORT = 30301
# PORT = 8081
ALERT_PORT = 30302
ALERT_INTERVAL = 0.5

JPEG_QUALITY = 10
PACKET_PAYLOAD = 2800
WIDTH = 1280
HEIGHT = 720
FPS = 30

NTPHEAD = 50500
NTP_SAMPLES = 50
ENABLE_NTP = True

HEADER_STRUCT = struct.Struct("<iiii d d d d")
HEADER_SIZE = HEADER_STRUCT.size

global delay
global expTime

delay = 0
expTime = 0

global frame_image
global display_image
global last_alert_time
last_alert_time = 0.0


def now_time():
    return time.time() - expTime


def ensure_log_dir():
    log_dir = os.path.join(CURRENT_DIR, "latency_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def udp_send(image, socket_process, frame_id, t_capture, t_detect_end, log_writer):
    """
    :param image: image data to send
    :param socket_process: UDP socket
    :return:
    """
    t_encode_start = now_time()
    encoded_image = cv2.imencode(
        '.jpg',
        image,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )[1].tobytes()
    t_encode_end = now_time()
    # split into packets
    data_length = len(encoded_image)

    packets = [encoded_image[i: i + PACKET_PAYLOAD] for i in range(0, data_length, PACKET_PAYLOAD)]
    total_packets = len(packets)
    t_send_end = t_encode_end
    for index, packet in enumerate(packets):
        packet_index = index + 1
        is_last = index == total_packets - 1
        if is_last:
            t_send_end = now_time()
            header = HEADER_STRUCT.pack(
                data_length,
                0,
                frame_id,
                JPEG_QUALITY,
                t_capture,
                t_detect_end,
                t_encode_end,
                t_send_end,
            )
            send_data = header + packet
            socket_process.sendto(send_data, (IP_ADDRESS, PORT))
        else:
            header = HEADER_STRUCT.pack(
                data_length,
                packet_index,
                frame_id,
                JPEG_QUALITY,
                t_capture,
                t_detect_end,
                t_encode_end,
                t_send_end,
            )
            send_data = header + packet
            socket_process.sendto(send_data, (IP_ADDRESS, PORT))
            time.sleep(0.001)
    if log_writer:
        log_writer.writerow(
            [
                frame_id,
                t_capture,
                t_detect_end,
                t_encode_end,
                t_send_end,
                data_length,
                JPEG_QUALITY,
            ]
        )


def send_thread(local_img, sock, frame_id, t_capture, t_detect_end, log_writer):
    udp_send(local_img, sock, frame_id, t_capture, t_detect_end, log_writer)


def maybe_send_alert(sock, detected_labels, frame_id):
    global last_alert_time
    if not detected_labels:
        return
    if "person" not in detected_labels:
        return
    now_ts = time.time()
    if now_ts - last_alert_time < ALERT_INTERVAL:
        return
    last_alert_time = now_ts
    payload = {
        "event": "detected",
        "labels": detected_labels,
        "frame_id": frame_id,
        "ts": now_ts,
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), (IP_ADDRESS, ALERT_PORT))


def resolve_thread(sock):
    while True:
        local_img = frame_image.copy()
        anomaly_detect_img, _ = anomaly_detect(local_img, visualize=False)
        global display_image
        display_image = anomaly_detect_img.copy()
        third_thread = threading.Thread(target=send_thread, args=(anomaly_detect_img, sock))
        third_thread.start()


def ntp_sync(sock):
    global delay
    global expTime
    if not ENABLE_NTP:
        return
    delay = 0
    expTime = 0
    for i in range(NTP_SAMPLES):
        now_time_local = time.time()
        send_data = struct.pack('i', NTPHEAD) + struct.pack('d', now_time_local)
        sock.sendto(send_data, (IP_ADDRESS, PORT))
        try:
            data, addr = sock.recvfrom(28)
            time_4 = time.time()
        except Exception:
            print('NTP recv timeout')
            raise IOError
        else:
            time_1 = struct.unpack('d', data[4:12])[0]
            time_2 = struct.unpack('d', data[12:20])[0]
            time_3 = struct.unpack('d', data[20:28])[0]
            delay += ((time_4 - time_1) - (time_3 - time_2)) / 2
            expTime += ((time_4 - time_3) - (time_2 - time_1)) / 2
    delay = delay / NTP_SAMPLES
    expTime = expTime / NTP_SAMPLES
    print(f"NTP delay: {delay:.6f}, expTime: {expTime:.6f}")


def main_thread(ip: str, port: int):
    global IP_ADDRESS
    global PORT
    IP_ADDRESS = ip
    PORT = port

    print("Start Initializing Socket Process")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # -------------NTP-Begin------------#
    ntp_sync(sock)
    # -------------NTP--End-------------#

    print("Start Initializing Video Camera")
    cameraCapture = cv2.VideoCapture(0)

    cv2.namedWindow('Car Detected', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Detected', WIDTH, HEIGHT)

    success, frame = cameraCapture.read()

    global frame_image
    frame_image = np.zeros((HEIGHT, WIDTH, 3))
    frame_image = cv2.resize(frame, (WIDTH, HEIGHT))
    print("Main Thread Frame:", end="")

    log_dir = ensure_log_dir()
    log_path = os.path.join(log_dir, "rsu_latency.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        log_writer.writerow(
            [
                "frame_id",
                "t_capture",
                "t_detect_end",
                "t_encode_end",
                "t_send_end",
                "encoded_bytes",
                "jpeg_quality",
            ]
        )

    frame_id = 0
    while True:
        success, frame = cameraCapture.read()
        frame_image = frame

        t_capture = now_time()
        anomaly_detect_img, detected_labels = anomaly_detect(frame_image, visualize=False)
        t_detect_end = now_time()

        frame_id += 1
        third_thread = threading.Thread(
            target=send_thread,
            args=(anomaly_detect_img, sock, frame_id, t_capture, t_detect_end, log_writer),
        )
        third_thread.start()
        third_thread.join()
        maybe_send_alert(sock, detected_labels, frame_id)

        cv2.imshow('Car Detected', anomaly_detect_img[..., ::-1])

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break

    log_file.close()
    cameraCapture.release()
    sock.close()


if __name__ == '__main__':
    main_thread(IP_ADDRESS, PORT)
