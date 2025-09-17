import asyncio
import json
import struct
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()

# Load model (sesuaikan path / device jika perlu)
MODEL_PATH = "yolo11m.pt"  # ganti sesuai model Anda
DEVICE = None  # None -> ultralytics auto (GPU jika tersedia)
print("Loading model...")
model = YOLO(MODEL_PATH)  # menggunakan default device atau set environment CUDA_VISIBLE_DEVICES
print("Model loaded.")

# Utilities


def parse_binary_payload(payload: bytes) -> (Dict[str, Any], bytes):
    """
    Payload format:
      4-byte big-endian unsigned int = header_length (N)
      next N bytes = header JSON (utf-8)
      remaining bytes = jpeg bytes
    Returns (header_dict, jpeg_bytes)
    """
    if len(payload) < 4:
        raise ValueError("Payload too short to contain header length")
    header_len = struct.unpack(">I", payload[:4])[0]
    if 4 + header_len > len(payload):
        raise ValueError("Payload too short for declared header length")
    header_bytes = payload[4 : 4 + header_len]
    jpeg_bytes = payload[4 + header_len :]
    header = json.loads(header_bytes.decode("utf-8"))
    return header, jpeg_bytes


def decode_jpeg_bytes(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode JPEG bytes to BGR numpy array (cv2 format).
    Returns None if decode fails.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def build_detection_list(results, include_start_time: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Convert ultralytics results to list of detection dicts:
    [
      {"label": str, "confidence": float, "x": int, "y": int, "w": int, "h": int, "start_time": float},
      ...
    ]
    """
    detections: List[Dict[str, Any]] = []
    for r in results:
        # r.boxes may be empty; handle both CPU tensors & numpy depending on version
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        try:
            # Try extraction via .xyxy / .conf / .cls (tensor)
            xyxy = r.boxes.xyxy.cpu().numpy()  # shape (N,4)
            confs = r.boxes.conf.cpu().numpy()  # (N,)
            classes = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        except Exception:
            # Fallback: try attributes as numpy already
            try:
                xyxy = np.array(r.boxes.xyxy)
                confs = np.array(r.boxes.conf)
                classes = np.array(r.boxes.cls).astype(int)
            except Exception:
                # Unable to parse boxes
                continue

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            label = model.names[int(cls)] if hasattr(model, "names") else str(int(cls))
            det = {
                "label": label,
                "confidence": float(conf),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
            if include_start_time is not None:
                det["start_time"] = include_start_time
            detections.append(det)
    return detections


async def run_inference(frame_bgr: np.ndarray) -> Any:
    """
    Run model inference in a thread (non-blocking).
    Convert BGR->RGB first as ultralytics expects RGB.
    Returns model results object.
    """
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Run in thread to avoid blocking event loop
    results = await asyncio.to_thread(model, frame_rgb)
    return results


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_addr = websocket.client.host if websocket.client else "unknown"
    print(f"Client connected: {client_addr}")

    try:
        while True:
            msg = await websocket.receive()

            # websocket.receive() returns dict with 'type' and either 'text' or 'bytes'
            data_type = msg.get("type")
            frame = None
            header = {}
            start_time_header = None

            if data_type == "websocket.receive":
                # prefer bytes if present
                if "bytes" in msg and msg["bytes"] is not None:
                    payload = msg["bytes"]
                    try:
                        header, jpeg_bytes = parse_binary_payload(payload)
                        start_time_header = header.get("start_time")
                        frame = decode_jpeg_bytes(jpeg_bytes)
                    except Exception as e:
                        # If parsing fails, try treat payload as plain jpeg bytes (no header)
                        try:
                            frame = decode_jpeg_bytes(payload)
                        except Exception:
                            print(f"Failed to parse binary payload: {e}")
                            frame = None

                elif "text" in msg and msg["text"] is not None:
                    text = msg["text"]
                    # Fallback: old client sends base64 text
                    try:
                        import base64

                        jpeg_bytes = base64.b64decode(text)
                        frame = decode_jpeg_bytes(jpeg_bytes)
                    except Exception as e:
                        print(f"Failed to decode text frame: {e}")
                        frame = None
                else:
                    # Unexpected: no content
                    continue
            else:
                # other events: ignore
                continue

            if frame is None:
                # nothing to do
                # optionally send empty detections
                await websocket.send_text(json.dumps([]))
                continue

            # Optionally display frame on server for debug (can be disabled)
            # cv2.imshow("Server View", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Run inference (async)
            try:
                results = await run_inference(frame)
            except Exception as e:
                print(f"Inference error: {e}")
                await websocket.send_text(json.dumps([]))
                continue

            # Build detections list and include start_time if available
            detections = build_detection_list(results, include_start_time=start_time_header or time.time())

            # send back JSON text
            try:
                await websocket.send_text(json.dumps(detections))
            except Exception as e:
                print(f"Failed to send detections: {e}")
                break

    except WebSocketDisconnect:
        print(f"Client disconnected: {client_addr}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        # cv2.destroyAllWindows()
        print("Connection cleanup done.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")