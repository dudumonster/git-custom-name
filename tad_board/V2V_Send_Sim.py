# -*- coding: UTF-8 -*-
import argparse
import json
import socket
import time


def build_message(text, seq):
    return {
        "event": "v2v_alert",
        "seq": seq,
        "msg": text,
        "ts": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="V2V warning simulator (car2 -> car1)")
    parser.add_argument("--target-ip", default="10.247.51.238", help="Car1 IP address")
    parser.add_argument("--port", type=int, default=30402, help="UDP port on car1")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between messages")
    parser.add_argument("--mode", default="cycle", choices=["cycle", "random"], help="Message mode")
    args = parser.parse_args()

    messages = [
        "前方有来车，请减速慢行",
        "拐弯处有行人，请减速慢行",
    ]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    try:
        while True:
            if args.mode == "random":
                msg = messages[int(time.time()) % len(messages)]
            else:
                msg = messages[seq % len(messages)]
            payload = build_message(msg, seq)
            sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), (args.target_ip, args.port))
            print(f"send -> {args.target_ip}:{args.port} | {msg}")
            seq += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
