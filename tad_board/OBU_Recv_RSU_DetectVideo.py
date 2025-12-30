# -*- coding: UTF-8 -*-
import sys
import time

import cv2
import struct
import socket
import numpy as np
import threading
import gc
import os
import csv
import json

from queue import Queue

# IP_ADDRESS_BIND = '192.168.62.117'
IP_ADDRESS_BIND = "10.247.51.238"
PORT_BIND = 30301
# PORT_BIND = 8081

# IP_ADDRESS = '192.168.62.199'
IP_ADDRESS = "10.247.51.2"
PORT = 30300
# PORT = 8080

PACKET_PAYLOAD = 2800
WIDTH = 640
HEIGHT = 480

NTPHEAD = 50500
NTP_SAMPLES = 50
ENABLE_NTP = True
ALERT_PORT = 30302
SHOW_PACKET_LOG = False
SHOW_VIDEO = True

HEADER_STRUCT = struct.Struct("<iiii d d d d")
HEADER_SIZE = HEADER_STRUCT.size
PACKET_READ_SIZE = HEADER_SIZE + PACKET_PAYLOAD


def print_log(is_error: bool = False, content: str = " "):
    content = content + '\n'
    if is_error:
        sys.stderr.write(content)
    else:
        sys.stdout.write(content)


def ensure_log_dir():
    log_dir = os.path.join(os.path.dirname(__file__), "latency_logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def display(end_event, data):
    if not SHOW_VIDEO:
        return
    cv2.imshow('Recv Video', data[..., ::-1])
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        end_event.set()


def receive(sock, queue: Queue, end_event: threading.Event, log_writer):
    image_total = b''
    img_packet_dict = {}
    total_size = 0
    last_header = None
    while not end_event.is_set():
        try:
            data, addr = sock.recvfrom(PACKET_READ_SIZE)
        except Exception:
            print_log(content='Recv timeout')
            raise IOError
        else:
            header = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
            fhead_size = header[0]
            count = header[1]
            frame_id = header[2]
            jpeg_quality = header[3]
            t_capture = header[4]
            t_detect_end = header[5]
            t_encode_end = header[6]
            t_send_end = header[7]
            img_packet = data[HEADER_SIZE:]

            img_packet_dict[count] = img_packet
            total_size += len(img_packet)
            if count == 0:
                last_header = header

            now_time = time.time()
            if SHOW_PACKET_LOG:
                print_log(content=f'Fhead:{fhead_size}, Count:{count}, '
                                  f'Recv:{len(img_packet)}, SumRecv:{total_size}')
            if count == 0:
                if total_size == fhead_size:
                    end_packet = img_packet_dict[count]
                    del img_packet_dict[count]
                    for i in sorted(img_packet_dict):
                        image_total += img_packet_dict[i]
                    image_total += end_packet
                    nparr = np.frombuffer(image_total, np.uint8)
                    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    t_recv_complete = time.time()
                    if SHOW_PACKET_LOG:
                        print_log(content='Add a new frame')
                    if SHOW_VIDEO:
                        queue.put(img_decode)
                    image_total = b''
                    img_packet_dict.clear()
                    total_size = 0
                    gc.collect()

                    if last_header:
                        compute_delay = t_detect_end - t_capture
                        encode_delay = t_encode_end - t_detect_end
                        tx_delay = t_send_end - t_encode_end
                        prop_delay = t_recv_complete - t_send_end
                        total_delay = t_recv_complete - t_capture
                        if log_writer:
                            log_writer.writerow(
                                [
                                    frame_id,
                                    t_capture,
                                    t_detect_end,
                                    t_encode_end,
                                    t_send_end,
                                    t_recv_complete,
                                    compute_delay,
                                    encode_delay,
                                    tx_delay,
                                    prop_delay,
                                    total_delay,
                                    jpeg_quality,
                                ]
                            )
                    last_header = None
                else:
                    image_total = b''
                    img_packet_dict.clear()
                    total_size = 0


def display_thread_wrapper(queue: Queue, end_event: threading.Event, wait_timeout: int, total_timeout: int):
    total_wait_time = 0
    if SHOW_VIDEO:
        cv2.namedWindow('Recv Video', cv2.WINDOW_AUTOSIZE)
    wait_path = os.path.join(os.path.dirname(__file__), "waitImg.png")
    wait_img = cv2.imread(wait_path)
    if wait_img is None:
        wait_bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    else:
        wait_bg = cv2.resize(wait_img, (WIDTH, HEIGHT))
    while not end_event.is_set():
        if total_wait_time >= total_timeout:
            print_log(is_error=True, content='Display wait timeout, quit!')
            break
        try:
            data = queue.get(block=True, timeout=wait_timeout)
        except Exception:
            if SHOW_PACKET_LOG:
                print_log(content='No data in display queue\n')
            display(end_event, wait_bg)
            total_wait_time += wait_timeout
        else:
            display(end_event, data)
            total_wait_time = 0
    print_log(content='Display thread quit')


def receive_thread_wrapper(queue: Queue, end_event: threading.Event, timeout: int, total_timeout: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP_ADDRESS_BIND, PORT_BIND))

    #-------------NTP-Begin------------#
    if ENABLE_NTP:
        count = 0
        while count < NTP_SAMPLES:
            try:
                data, addr = sock.recvfrom(28)
                time_2 = time.time()
            except Exception:
                print('Recv timeout')
                raise IOError
            else:
                time_1 = struct.unpack('d', data[4:12])[0]
                time_3 = time.time()
                send_data = struct.pack('i', NTPHEAD) + struct.pack('d', time_1) + struct.pack('d', time_2) + struct.pack('d', time_3)
                sock.sendto(send_data, addr)
                count += 1
    #-------------NTP--End-------------#

    sock.settimeout(timeout)
    print("Bind Up on 30301")
    print('Start Receiving ...')

    log_dir = ensure_log_dir()
    log_path = os.path.join(log_dir, "obu_latency.csv")
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
                "t_recv_complete",
                "compute_delay",
                "encode_delay",
                "tx_delay",
                "prop_delay",
                "total_delay",
                "jpeg_quality",
            ]
        )

    total_wait_time = 0
    while not end_event.is_set():
        if total_wait_time >= total_timeout:
            print_log(is_error=True, content='Receive wait timeout, quit!')
            break
        try:
            receive(sock, queue, end_event, log_writer)
        except IOError:
            total_wait_time += timeout
        else:
            total_wait_time = 0

    log_file.close()
    sock.close()
    print_log(content='Receive thread quit')


def alert_thread_wrapper(end_event: threading.Event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP_ADDRESS_BIND, ALERT_PORT))
    print(f"Alert listener on {IP_ADDRESS_BIND}:{ALERT_PORT}")
    while not end_event.is_set():
        try:
            data, addr = sock.recvfrom(2048)
        except Exception:
            continue
        else:
            message = data.decode("utf-8", errors="ignore")
            try:
                payload = json.loads(message)
                labels = payload.get("labels", [])
                frame_id = payload.get("frame_id", "unknown")
                print(f"检测到目标: {labels} (frame {frame_id})")
            except Exception:
                print(message)
    sock.close()


def main_thread(ip: str, port: int):
    global IP_ADDRESS_BIND
    global PORT_BIND
    IP_ADDRESS_BIND = ip
    PORT_BIND = port

    timeout = 1
    max_wait_time = 100
    kill_event = threading.Event()
    display_queue = Queue()
    display_thread = threading.Thread(
        target=display_thread_wrapper,
        args=(display_queue, kill_event, timeout, timeout * max_wait_time)
    )
    receive_thread = threading.Thread(
        target=receive_thread_wrapper,
        args=(display_queue, kill_event, timeout, timeout * max_wait_time)
    )
    alert_thread = threading.Thread(
        target=alert_thread_wrapper,
        args=(kill_event,)
    )

    receive_thread.start()
    display_thread.start()
    alert_thread.start()

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        kill_event.set()

    display_thread.join()
    receive_thread.join()
    alert_thread.join()


if __name__ == '__main__':
    main_thread(IP_ADDRESS_BIND, PORT_BIND)
