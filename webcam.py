"""
Live webcam client for the YOLOv8 Object Detection API.

1. Start the API first:   fastapi dev main.py
2. Then run this client:  python webcam.py
3. Press 'q' to quit, 's' to save a screenshot to screenshots/webcam_detection.png

Optional arguments:
    python webcam.py --url http://localhost:8000 --confidence 0.5 --classes person,cup --camera 0
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests


# def parse_args():
#     parser.add_argument("--source", default="", help="Video file path or URL")
#     parser = argparse.ArgumentParser(description="YOLOv8 webcam client")
#     parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
#     parser.add_argument("--confidence", type=float, default=0.4, help="Min confidence (0-1)")
#     parser.add_argument("--classes", default="", help="Comma-separated class names, empty = all")
#     parser.add_argument("--camera", type=int, default=0, help="Webcam index")
#     parser.add_argument("--width", type=int, default=640, help="Frame width sent to the API")
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 webcam client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--confidence", type=float, default=0.4, help="Min confidence (0-1)")
    parser.add_argument("--classes", default="", help="Comma-separated class names, empty = all")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=640, help="Frame width sent to the API")
    parser.add_argument("--source", default="", help="Video file path or URL")
    return parser.parse_args()


def main():
    args = parse_args()
    api_url = f"{args.url.rstrip('/')}/detect/image"

    # cap = cv2.VideoCapture(args.camera)
    # cap = cv2.VideoCapture(args.source if getattr(args, "source", "") else args.camera, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(args.source) if args.source else cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Could not open webcam #{args.camera}")
        return

    session = requests.Session()  # reuse the HTTP connection for speed
    prev_time = time.time()
    print("Webcam started. Press 'q' to quit, 's' to save a screenshot.")

    while True:
        ok, frame = cap.read()
        if not ok:
            # print("Could not read frame from webcam")
            print("End of video or camera disconnected")
            break

        # Resize big frames to keep the request small and fast
        h, w = frame.shape[:2]
        if w > args.width:
            frame = cv2.resize(frame, (args.width, int(h * args.width / w)))

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            continue

        status_text = "API offline"
        try:
            response = session.post(
                api_url,
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                params={"confidence": args.confidence, "classes": args.classes},
                timeout=5,
            )
            if response.status_code == 200:
                array = np.frombuffer(response.content, np.uint8)
                annotated = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if annotated is None:
                    annotated = frame
                counts = json.loads(response.headers.get("X-Detections", "{}"))
                status_text = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "no objects"
            else:
                annotated = frame
                status_text = f"API error {response.status_code}"
        except requests.RequestException:
            annotated = frame

        # FPS overlay
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f} | {status_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("YOLOv8 Webcam", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            Path("screenshots").mkdir(exist_ok=True)
            cv2.imwrite("screenshots/webcam_detection.png", annotated)
            print("Saved screenshots/webcam_detection.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
