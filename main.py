"""
YOLOv8 Object Detection API
----------------------------
FastAPI backend that runs a pretrained YOLOv8 (COCO) model on uploaded
images and short videos and returns the annotated result.

Run:
    fastapi dev main.py
Docs:
    http://localhost:8000/docs
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_NAME = "yolov8n.pt"
MAX_IMAGE_BYTES = 10 * 1024 * 1024    # 10 MB
MAX_VIDEO_BYTES = 100 * 1024 * 1024   # 100 MB
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description=(
        "Real-time object detection with a pretrained YOLOv8 model. "
        "Upload an image or a short video and get the annotated result back."
    ),
    version="1.0.0",
)

# Load the model ONCE at startup (downloads automatically on first use).
model = YOLO(MODEL_NAME)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_class_ids(classes: str) -> Optional[List[int]]:
    """
    Convert a comma-separated list of class names (e.g. "person,car")
    into the list of class IDs expected by Ultralytics.

    Returns None when no classes are requested (= detect everything).
    Raises HTTP 400 if any requested name is not a COCO class.
    """
    selected = [name.strip().lower() for name in classes.split(",") if name.strip()]
    if not selected:
        return None

    name_to_id = {name.lower(): class_id for class_id, name in model.names.items()}
    unknown = [name for name in selected if name not in name_to_id]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown class name(s): {', '.join(unknown)}. "
                   f"See GET /classes for valid names.",
        )
    return sorted({name_to_id[name] for name in selected})


def count_detections(result) -> Dict[str, int]:
    """Count detected objects per class using results[0].boxes.cls."""
    counts: Dict[str, int] = {}
    boxes = result.boxes
    if boxes is not None:
        for class_id in boxes.cls.tolist():
            name = model.names[int(class_id)]
            counts[name] = counts.get(name, 0) + 1
    return counts


def detect_frame(frame: np.ndarray, confidence: float, class_ids: Optional[List[int]]):
    """Run YOLO on one BGR frame and return (annotated_frame, counts)."""
    results = model(frame, conf=confidence, classes=class_ids, verbose=False)
    result = results[0]
    # result.boxes.xyxy -> [xmin, ymin, xmax, ymax]
    # result.boxes.cls  -> class IDs
    # result.boxes.conf -> confidence scores
    annotated = result.plot()  # draws boxes, labels and confidence scores
    return annotated, count_detections(result)


def process_video(input_path: Path, output_path: Path,
                  confidence: float, class_ids: Optional[List[int]]) -> Dict[str, int]:
    """Run detection on every frame of a video and write the annotated video."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Invalid video file")

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frames = 0
    max_counts: Dict[str, int] = {}  # max objects of each class seen in a single frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, counts = detect_frame(frame, confidence, class_ids)
            writer.write(annotated)
            frames += 1
            for name, n in counts.items():
                max_counts[name] = max(max_counts.get(name, 0), n)
    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raise HTTPException(status_code=400, detail="Invalid video file (no frames)")

    return {"frames": frames, "max_per_frame": max_counts}


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def health_check():
    """Health check: returns status and model information."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "task": model.task,
        "num_classes": len(model.names),
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    """Return the 80 COCO class names available in the pretrained model."""
    return {"count": len(model.names), "classes": list(model.names.values())}


@app.post(
    "/detect/image",
    tags=["Detection"],
    response_class=StreamingResponse,
    responses={200: {"content": {"image/jpeg": {}},
                     "description": "Annotated JPEG. Per-class counts in the X-Detections header."},
               400: {"description": "Invalid image or class name"}},
)
async def detect_image(
    file: UploadFile = File(..., description="Image file (jpg, png, ...)"),
    confidence: float = Query(0.4, ge=0.0, le=1.0, description="Minimum confidence score"),
    classes: str = Query("", description="Comma-separated class names, e.g. person,car. Empty = all"),
):
    """Detect objects in an image and return the annotated JPEG."""
    class_ids = get_class_ids(classes)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Invalid image file (empty upload)")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 10 MB)")

    array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Inference is CPU/GPU bound - run it outside the event loop.
    annotated, counts = await run_in_threadpool(detect_frame, frame, confidence, class_ids)

    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode output image")

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={
            "X-Detections": json.dumps(counts),
            "Access-Control-Expose-Headers": "X-Detections",
        },
    )


@app.post(
    "/detect/video",
    tags=["Detection"],
    response_class=Response,
    responses={200: {"content": {"video/mp4": {}},
                     "description": "Annotated MP4 video"},
               400: {"description": "Invalid video or class name"}},
)
async def detect_video(
    file: UploadFile = File(..., description="Short video file (mp4, avi, mov, ...)"),
    confidence: float = Query(0.4, ge=0.0, le=1.0, description="Minimum confidence score"),
    classes: str = Query("", description="Comma-separated class names, e.g. person,car. Empty = all"),
):
    """Detect objects in a short video frame by frame and return the annotated video."""
    class_ids = get_class_ids(classes)

    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {suffix}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Invalid video file (empty upload)")
    if len(contents) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Video is too large (max 100 MB)")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / f"input{suffix}"
        output_path = temp_dir / "output.mp4"
        input_path.write_bytes(contents)

        summary = await run_in_threadpool(
            process_video, input_path, output_path, confidence, class_ids
        )
        video_bytes = output_path.read_bytes()

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": "attachment; filename=detected.mp4",
            "X-Detections": json.dumps(summary),
            "Access-Control-Expose-Headers": "X-Detections",
        },
    )
