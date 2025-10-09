import asyncio
import json
import logging
import struct
import time
from typing import Any, Dict, List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import mediapipe as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Vision Server", description="Production-ready AI Vision Server with YOLO and MediaPipe support")

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model types
MODEL_TYPE = Literal["yolo", "mediapipe"]

# Global thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=8)

# Model registry for different AI models
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_types = {}
        self.load_default_models()
    
    def load_default_models(self):
        """Load default models at startup"""
        # Load YOLO models
        try:
            self.models["yolo_yolo11m"] = YOLO("yolo11m.pt")
            self.model_types["yolo_yolo11m"] = "yolo"
            logger.info("YOLOv11m model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv11m model: {e}")
        
        try:
            self.models["yolo_yolov8n"] = YOLO("yolov8n.pt")
            self.model_types["yolo_yolov8n"] = "yolo"
            logger.info("YOLOv8n model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8n model: {e}")
        
        try:
            self.models["yolo_yolov8m"] = YOLO("yolov8m.pt")
            self.model_types["yolo_yolov8m"] = "yolo"
            logger.info("YOLOv8m model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8m model: {e}")
        
        # Load MediaPipe models
        try:
            self.models["mediapipe_pose"] = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.model_types["mediapipe_pose"] = "mediapipe"
            logger.info("MediaPipe pose model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe pose model: {e}")
        
        try:
            self.models["mediapipe_hands"] = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.model_types["mediapipe_hands"] = "mediapipe"
            logger.info("MediaPipe hands model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe hands model: {e}")
        
        try:
            self.models["mediapipe_objects"] = mp.solutions.objectron.Objectron(
                static_image_mode=False,
                max_num_objects=5,
                model_name='Shoe'
            )
            self.model_types["mediapipe_objects"] = "mediapipe"
            logger.info("MediaPipe objectron model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe objectron model: {e}")
        
        try:
            self.models["mediapipe_face"] = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            self.model_types["mediapipe_face"] = "mediapipe"
            logger.info("MediaPipe face detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe face detection model: {e}")
    
    def get_model(self, model_name: str):
        """Get model by name"""
        return self.models.get(model_name)
    
    def get_model_type(self, model_name: str):
        """Get model type by name"""
        return self.model_types.get(model_name)
    
    def list_models(self):
        """Get list of available models"""
        return list(self.models.keys())

model_manager = ModelManager()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}
        self.connection_lock = threading.Lock()
    
    def add_connection(self, websocket: WebSocket, client_id: str, model_type: MODEL_TYPE, model_name: str):
        with self.connection_lock:
            self.active_connections[client_id] = {
                "websocket": websocket,
                "model_type": model_type,
                "model_name": model_name,
                "connected_at": time.time()
            }
    
    def remove_connection(self, client_id: str):
        with self.connection_lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
    
    def get_client_info(self, client_id: str):
        with self.connection_lock:
            return self.active_connections.get(client_id)
    
    def get_active_count(self):
        with self.connection_lock:
            return len(self.active_connections)
    
    def list_connections(self):
        with self.connection_lock:
            return list(self.active_connections.keys())

connection_manager = ConnectionManager()

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


def convert_yolo_results_to_detections(results, include_start_time: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Convert YOLO results to list of detection dicts.
    """
    detections: List[Dict[str, Any]] = []
    for r in results:
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
            # Get label from model names if available
            model_obj = None
            # Find the model object that was used (this is a simplified approach)
            for model in model_manager.models.values():
                if hasattr(model, 'names') and int(cls) < len(model.names):
                    label = model.names[int(cls)]
                    break
            else:
                label = str(int(cls))
                
            det = {
                "label": label,
                "confidence": float(conf),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "type": "object_detection"
            }
            if include_start_time is not None:
                det["start_time"] = include_start_time
            detections.append(det)
    return detections


def convert_mediapipe_results_to_detections(results, model_name: str, include_start_time: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Convert MediaPipe results to list of detection dicts based on the model type.
    """
    model_obj = model_manager.get_model(model_name)
    if model_obj is None:
        logger.error(f"Model {model_name} not found in model manager")
        return []
    
    # Determine the model type based on the model name
    if 'pose' in model_name:
        model_type = 'pose'
    elif 'hands' in model_name:
        model_type = 'hands' 
    elif 'objects' in model_name:
        model_type = 'objectron'
    elif 'face' in model_name:
        model_type = 'face'
    else:
        logger.error(f"Unknown MediaPipe model type for {model_name}")
        return []
    
    detections: List[Dict[str, Any]] = []
    
    if model_type == 'pose' and results.pose_landmarks:
        # Process pose landmarks
        landmarks = results.pose_landmarks.landmark
        for i, landmark in enumerate(landmarks):
            det = {
                "label": f"pose_keypoint_{i}",
                "confidence": landmark.visibility if hasattr(landmark, 'visibility') else 0.8,
                "x": int(landmark.x * 640),  # Assuming 640x480 frame, need to adjust based on actual frame size
                "y": int(landmark.y * 480),
                "w": 2,  # Small width for keypoint
                "h": 2,  # Small height for keypoint
                "type": "pose_landmark"
            }
            if include_start_time is not None:
                det["start_time"] = include_start_time
            detections.append(det)

    elif model_type == 'hands' and results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for j, landmark in enumerate(hand_landmarks.landmark):
                det = {
                    "label": f"hand_{i}_keypoint_{j}",
                    "confidence": 0.8,  # MediaPipe doesn't provide confidence for hand landmarks
                    "x": int(landmark.x * 640),  # Adjust based on frame size
                    "y": int(landmark.y * 480),
                    "w": 2,
                    "h": 2,
                    "type": "hand_landmark"
                }
                if include_start_time is not None:
                    det["start_time"] = include_start_time
                detections.append(det)

    elif model_type == 'objectron' and results.detected_objects:
        for detected_object in results.detected_objects:
            # Extract bounding box from the detected object
            if hasattr(detected_object, 'bounding_box'):
                bbox = detected_object.bounding_box
                det = {
                    "label": detected_object.category_name if hasattr(detected_object, 'category_name') else "object",
                    "confidence": detected_object.score if hasattr(detected_object, 'score') else 0.7,
                    "x": int(bbox.origin_x),
                    "y": int(bbox.origin_y),
                    "w": int(bbox.width),
                    "h": int(bbox.height),
                    "type": "object_detection"
                }
                if include_start_time is not None:
                    det["start_time"] = include_start_time
                detections.append(det)

    elif model_type == 'face' and results.detections:
        for i, detection in enumerate(results.detections):
            if hasattr(detection, 'location_data'):
                bbox = detection.location_data.relative_bounding_box
                det = {
                    "label": "face",
                    "confidence": detection.score[0] if hasattr(detection, 'score') and len(detection.score) > 0 else 0.7,
                    "x": int(bbox.xmin * 640),  # Adjust based on frame size
                    "y": int(bbox.ymin * 480),
                    "w": int(bbox.width * 640),
                    "h": int(bbox.height * 480),
                    "type": "face_detection"
                }
                if include_start_time is not None:
                    det["start_time"] = include_start_time
                detections.append(det)
    
    return detections


async def run_yolo_inference(frame_bgr: np.ndarray, model_name: str) -> Any:
    """
    Run YOLO model inference in a thread (non-blocking).
    Convert BGR->RGB first as ultralytics expects RGB.
    Returns model results object.
    """
    model = model_manager.get_model(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found")
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Run in thread to avoid blocking event loop
    results = await asyncio.to_thread(model, frame_rgb)
    return results


async def run_mediapipe_inference(frame_bgr: np.ndarray, model_name: str) -> Any:
    """
    Run MediaPipe model inference in a thread (non-blocking).
    Returns model results object.
    """
    model_data = model_manager.get_model(model_name)
    if model_data is None or 'model' not in model_data:
        raise ValueError(f"Model {model_name} not found")
    
    model = model_data['model']
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Run MediaPipe inference (the model object itself is used to process)
    results = await asyncio.to_thread(model.process, frame_rgb)
        
    return results


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    
    # Wait for model selection message
    try:
        init_msg = await websocket.receive_text()
        init_data = json.loads(init_msg)
        
        model_type = init_data.get("model_type", "yolo")
        model_name = init_data.get("model_name", "yolo_yolo11m")
        
        if model_type not in ["yolo", "mediapipe"]:
            await websocket.send_text(json.dumps({
                "error": "Invalid model type. Use 'yolo' or 'mediapipe'"
            }))
            await websocket.close()
            return
            
        if model_name not in model_manager.list_models():
            available_models = [m for m in model_manager.list_models() if m.startswith(model_type)]
            await websocket.send_text(json.dumps({
                "error": f"Model {model_name} not found. Available {model_type} models: {available_models}"
            }))
            await websocket.close()
            return
        
        # Add connection to manager
        connection_manager.add_connection(websocket, client_id, model_type, model_name)
        logger.info(f"Client {client_id} connected with model {model_name} (type: {model_type})")
        logger.info(f"Active connections: {connection_manager.get_active_count()}")
        
        # Send confirmation to client
        await websocket.send_text(json.dumps({
            "status": "connected",
            "model": model_name,
            "type": model_type
        }))
        
    except Exception as e:
        logger.error(f"Failed to initialize connection for {client_id}: {e}")
        await websocket.close()
        return

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
                            logger.error(f"Failed to parse binary payload: {e}")
                            frame = None

                elif "text" in msg and msg["text"] is not None:
                    text = msg["text"]
                    # Fallback: client sends JSON command or base64 image
                    try:
                        json_data = json.loads(text)
                        # Check if it's a command
                        if "command" in json_data:
                            command = json_data["command"]
                            if command == "list_models":
                                await websocket.send_text(json.dumps({
                                    "models": model_manager.list_models()
                                }))
                            continue
                    except json.JSONDecodeError:
                        # If not JSON, assume it's base64 encoded image
                        try:
                            import base64
                            jpeg_bytes = base64.b64decode(text)
                            frame = decode_jpeg_bytes(jpeg_bytes)
                        except Exception as e:
                            logger.error(f"Failed to decode text frame: {e}")
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

            # Get client's selected model
            client_info = connection_manager.get_client_info(client_id)
            if not client_info:
                # Client disconnected
                break
                
            selected_model_type = client_info["model_type"]
            selected_model_name = client_info["model_name"]

            # Run inference based on selected model type
            try:
                if selected_model_type == "yolo":
                    results = await run_yolo_inference(frame, selected_model_name)
                    detections = convert_yolo_results_to_detections(results, include_start_time=start_time_header or time.time())
                elif selected_model_type == "mediapipe":
                    results = await run_mediapipe_inference(frame, selected_model_name)
                    detections = convert_mediapipe_results_to_detections(results, selected_model_name, include_start_time=start_time_header or time.time())
                else:
                    detections = []
            except Exception as e:
                logger.error(f"Inference error for {client_id}: {e}")
                await websocket.send_text(json.dumps([]))
                continue

            # send back JSON text
            try:
                await websocket.send_text(json.dumps(detections))
            except Exception as e:
                logger.error(f"Failed to send detections to {client_id}: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        connection_manager.remove_connection(client_id)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"Connection cleanup done for {client_id}. Active connections: {connection_manager.get_active_count()}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.list_models()),
        "active_connections": connection_manager.get_active_count(),
        "model_names": model_manager.list_models()
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": model_manager.list_models(),
        "count": len(model_manager.list_models())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "production_server:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        workers=1,  # For GPU applications, often better to use 1 worker with multiple threads
        timeout_keep_alive=300
    )