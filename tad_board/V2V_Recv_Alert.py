# -*- coding: UTF-8 -*-
import argparse
import json
import socket
import time


def format_ts(ts):
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return time.strftime("%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="V2V warning receiver (car1 side)")
    parser.add_argument("--bind-ip", default="10.247.51.238", help="Local bind IP on car1")
    parser.add_argument("--port", type=int, default=30402, help="UDP listen port")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    print(f"V2V listener on {args.bind_ip}:{args.port}")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except Exception:
            continue
        message = data.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(message)
            text = payload.get("msg", message)
            ts = payload.get("ts", time.time())
            print(f"[{format_ts(ts)}] V2V: {text} (from {addr[0]})")
        except Exception:
            print(f"V2V: {message} (from {addr[0]})")


if __name__ == "__main__":
    main()
