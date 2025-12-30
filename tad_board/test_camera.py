import argparse
import time
import cv2


def open_capture(device, api=None):
    """Open a camera by index (int) or name/path (str)."""
    if api:
        return cv2.VideoCapture(device, getattr(cv2, api))
    # Try int index first; fall back to string
    try:
        idx = int(device)
        return cv2.VideoCapture(idx)
    except (TypeError, ValueError):
        return cv2.VideoCapture(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, help="OpenCV device index (int) or name")
    parser.add_argument("--api", default=None, help="Optional backend (e.g. DSHOW on Windows)")
    parser.add_argument("--width", type=int, default=1920, help="Requested frame width")
    parser.add_argument("--height", type=int, default=1080, help="Requested frame height")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS")
    args = parser.parse_args()

    cap = open_capture(args.device, args.api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise SystemExit(f"无法打开设备：{args.device}")

    print("按 q 退出；按 s 保存截图(camera_snapshot.jpg)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读帧失败，可能设备被占用或未就绪")
            break

        h, w = frame.shape[:2]
        print(f"\r当前分辨率：{w}x{h} ({time.strftime('%H:%M:%S')})", end="", flush=True)

        cv2.imshow("Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite("camera_snapshot.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
