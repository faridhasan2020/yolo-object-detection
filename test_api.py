"""
Integration tests for the YOLOv8 Object Detection API.

Start the server first:
    fastapi dev main.py

Then run either:
    python test_api.py
    pytest test_api.py -v
"""

import json
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
SAMPLES = Path(__file__).parent / "samples"
TEST_IMAGE = SAMPLES / "test_image.jpg"
TEST_VIDEO = SAMPLES / "test_video.mp4"
TIMEOUT = 300


def post_image(path=TEST_IMAGE, confidence=0.4, classes=""):
    with open(path, "rb") as file:
        return requests.post(
            f"{BASE_URL}/detect/image",
            files={"file": ("test.jpg", file, "image/jpeg")},
            params={"confidence": confidence, "classes": classes},
            timeout=TIMEOUT,
        )


# 1. Health check
def test_health_check():
    response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "yolov8n.pt"


# 2. Classes endpoint returns the 80 COCO classes
def test_classes():
    response = requests.get(f"{BASE_URL}/classes", timeout=TIMEOUT)
    assert response.status_code == 200
    classes = response.json()["classes"]
    assert len(classes) == 80
    for name in ("person", "car", "dog"):
        assert name in classes


# 3. Image detection returns a valid JPEG
def test_detect_image():
    response = post_image()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic bytes


# 4. X-Detections header is present and valid JSON
def test_detection_counts():
    response = post_image()
    assert response.status_code == 200
    assert "X-Detections" in response.headers
    counts = json.loads(response.headers["X-Detections"])
    assert isinstance(counts, dict)
    assert all(isinstance(v, int) and v > 0 for v in counts.values())
    print("Detections:", counts)


# 5. Class filter keeps only the requested class
def test_class_filter():
    response = post_image(classes="person")
    assert response.status_code == 200
    counts = json.loads(response.headers["X-Detections"])
    if counts:
        assert set(counts.keys()) == {"person"}
    print("Filtered detections:", counts)


# 6. Invalid image returns HTTP 400
def test_invalid_image():
    response = requests.post(
        f"{BASE_URL}/detect/image",
        files={"file": ("bad.jpg", b"this is not an image", "image/jpeg")},
        timeout=TIMEOUT,
    )
    assert response.status_code == 400


# Extra: unknown class name returns HTTP 400
def test_unknown_class():
    response = post_image(classes="unicorn")
    assert response.status_code == 400


# 7. Video detection returns an annotated video
def test_detect_video():
    with open(TEST_VIDEO, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/detect/video",
            files={"file": ("test_video.mp4", file, "video/mp4")},
            params={"confidence": 0.4, "classes": ""},
            timeout=TIMEOUT,
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert len(response.content) > 1000


# --------------------------------------------------------------------------
# Plain runner: python test_api.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_health_check,
        test_classes,
        test_detect_image,
        test_detection_counts,
        test_class_filter,
        test_invalid_image,
        test_unknown_class,
        test_detect_video,
    ]

    try:
        requests.get(f"{BASE_URL}/", timeout=5)
    except requests.RequestException:
        print(f"API is not running at {BASE_URL}. Start it with: fastapi dev main.py")
        sys.exit(1)

    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as error:
            print(f"FAIL  {test.__name__}  {error}")
        except Exception as error:  # noqa: BLE001
            print(f"ERROR {test.__name__}  {type(error).__name__}: {error}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
