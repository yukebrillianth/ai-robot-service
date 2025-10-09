import json
import struct
import threading
import time
from collections import deque
from typing import Literal, Dict, Any, Optional

import cv2
import numpy as np
from websocket import WebSocket, ABNF

# Type aliases
MODEL_TYPE = Literal["yolo", "mediapipe"]
YOLO_MODEL = Literal["yolo_yolo11m", "yolo_yolov8n", "yolo_yolov8m"]
MEDIAPIPE_MODEL = Literal["mediapipe_pose", "mediapipe_hands", "mediapipe_objects", "mediapipe_face"]

class ProductionClient:
    def __init__(
        self,
        server_url: str,
        model_type: MODEL_TYPE = "yolo",
        model_name: str = "yolo_yolo11m",
        capture_index: int = 0,
        display_fps: int = 30,
        send_fps: int = 10,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 60,
        motion_threshold: float = 4.0,
        min_quality: int = 20,
        max_quality: int = 80,
        motion_downscale: tuple = (160, 120),
    ):
        self.server_url = server_url
        self.model_type = model_type
        self.model_name = model_name
        self.capture_index = capture_index
        self.display_fps = display_fps
        self.send_fps = send_fps
        self.resize_w = width
        self.resize_h = height
        self.jpeg_quality = jpeg_quality
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.motion_threshold = motion_threshold
        self.motion_downscale = motion_downscale

        self.ws = None
        self.cap = None
        self.detections = deque(maxlen=1)
        self.detections_lock = threading.Lock()
        self.latest_frame = None  # frame terakhir (resized) untuk dikirim (BGR)
        self.latest_frame_lock = threading.Lock()
        self.prev_sent_gray_small = None  # untuk motion detection di ukuran kecil
        self.running = False

        # stats untuk adaptasi kualitas
        self.send_time_ema = None
        self.ema_alpha = 0.2

        # Performance metrics
        self.frame_count = 0
        self.last_fps_update = time.time()

    def connect(self) -> bool:
        """Connect to the server and initialize model selection"""
        try:
            self.ws = WebSocket()
            self.ws.settimeout(10)  # longer timeout for initial connection
            self.ws.connect(self.server_url)
            
            # Send model selection after connection
            model_selection = {
                "model_type": self.model_type,
                "model_name": self.model_name
            }
            self.ws.send(json.dumps(model_selection))
            
            # Wait for server confirmation
            response = self.ws.recv()
            server_response = json.loads(response)
            if server_response.get("status") == "connected":
                print(f"Connected to server with model: {server_response['model']} (type: {server_response['type']})")
                return True
            else:
                print(f"Server connection failed: {server_response}")
                return False
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def start_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.capture_index)
        if not self.cap.isOpened():
            print("Failed to open camera")
            return False
        # Reduce buffer size for less latency
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return True

    def encode_frame_bytes(self, frame_bgr: np.ndarray, quality: int) -> Optional[bytes]:
        """Encode frame to JPEG bytes"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
        if not ok:
            return None
        return buffer.tobytes()

    def _build_payload(self, jpeg_bytes: bytes) -> bytes:
        """Build binary payload with header and JPEG data"""
        # Create metadata
        meta = {
            "width": self.resize_w,
            "height": self.resize_h,
            "format": "jpeg",
            "jpeg_quality": int(self.jpeg_quality),
            "start_time": time.time(),
            "model_type": self.model_type,
            "model_name": self.model_name
        }
        header = json.dumps(meta).encode('utf-8')
        prefix = struct.pack('>I', len(header))
        return prefix + header + jpeg_bytes

    def receive_detections(self):
        """Receive detection results from server in a separate thread"""
        while self.running:
            try:
                if self.ws and self.ws.connected:
                    try:
                        result = self.ws.recv()
                        detections = json.loads(result)
                        if (detections and isinstance(detections, list)
                                and len(detections) > 0 and 'start_time' in detections[0]):
                            now = time.time()
                            latency_ms = (now - detections[0]['start_time']) * 1000.0
                            print(f"Latency: {latency_ms:.2f} ms")
                        with self.detections_lock:
                            self.detections.append(detections)
                    except Exception:
                        # timeout / socket issue, continue
                        continue
            except Exception as e:
                if self.running:
                    print(f"Error receiving detections: {e}")
                time.sleep(0.5)

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection results on frame"""
        with self.detections_lock:
            if len(self.detections) == 0:
                return frame
            detections = self.detections[-1]
        
        if not isinstance(detections, list):
            return frame
            
        for det in detections:
            try:
                x, y, w, h = map(int, (det['x'], det['y'], det['w'], det['h']))
                label = det.get('label', 'unknown')
                conf = det.get('confidence', 0)
                det_type = det.get('type', 'object_detection')
                
                # Use different colors based on detection type
                if det_type == 'pose_landmark':
                    color = (255, 0, 0)  # Blue for pose landmarks
                elif det_type == 'hand_landmark':
                    color = (0, 255, 0)  # Green for hand landmarks
                elif det_type == 'face_detection':
                    color = (0, 0, 255)  # Red for faces
                else:
                    color = (0, 255, 0)  # Green for object detection
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame,
                            f"{label}:{conf:.2f}",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1)
            except (KeyError, ValueError):
                continue
        return frame

    def frame_should_send(self, frame_bgr: np.ndarray) -> bool:
        """Determine if frame should be sent based on motion detection"""
        # Compute motion on downscaled grayscale (very cheap)
        small = cv2.resize(frame_bgr, self.motion_downscale, interpolation=cv2.INTER_LINEAR)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_sent_gray_small is None:
            self.prev_sent_gray_small = gray_small
            return True
        diff = cv2.absdiff(self.prev_sent_gray_small, gray_small)
        mean_diff = float(diff.mean())
        percent = (mean_diff / 255.0) * 100.0
        if percent >= self.motion_threshold:
            self.prev_sent_gray_small = gray_small
            return True
        return False

    def adjust_quality_based_on_send(self, send_time: float):
        """Adjust JPEG quality based on network performance"""
        # EMA for send time
        if self.send_time_ema is None:
            self.send_time_ema = send_time
        else:
            self.send_time_ema = (1 - self.ema_alpha) * self.send_time_ema + self.ema_alpha * send_time
        # if sending is slow (more than 80% of send interval), decrease quality slightly
        target = 1.0 / max(1, self.send_fps)
        if self.send_time_ema > target * 0.8:
            self.jpeg_quality = max(self.min_quality, int(self.jpeg_quality * 0.9))
        else:
            # if it's still fast, increase slightly up to limit
            self.jpeg_quality = min(self.max_quality, int(self.jpeg_quality * 1.05))

    def sender_loop(self):
        """Send frames to server in a separate thread"""
        send_interval = 1.0 / max(1, self.send_fps)
        while self.running:
            start = time.time()
            frame = None
            with self.latest_frame_lock:
                if self.latest_frame is not None:
                    # use reference copy minimal
                    frame = self.latest_frame.copy()
            if frame is not None and self.ws and self.ws.connected:
                if self.frame_should_send(frame):
                    # encode to jpeg bytes
                    jpeg_bytes = self.encode_frame_bytes(frame, self.jpeg_quality)
                    if jpeg_bytes:
                        payload = self._build_payload(jpeg_bytes)
                        try:
                            send_start = time.time()
                            # send as binary
                            self.ws.send(payload, opcode=ABNF.OPCODE_BINARY)
                            send_time = time.time() - send_start
                            # adaptive quality
                            self.adjust_quality_based_on_send(send_time)
                        except Exception as e:
                            print(f"Failed to send frame: {e}")
            # Sleep remaining time
            elapsed = time.time() - start
            to_sleep = send_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def update_fps_display(self):
        """Update FPS counter display"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
            return f"FPS: {fps:.1f}"
        return ""

    def run(self):
        """Main execution loop"""
        if not self.connect():
            return
        if not self.start_camera():
            return
            
        self.running = True
        recv_thread = threading.Thread(target=self.receive_detections, daemon=True)
        send_thread = threading.Thread(target=self.sender_loop, daemon=True)
        recv_thread.start()
        send_thread.start()
        
        display_interval = 1.0 / max(1, self.display_fps)
        
        try:
            while self.running:
                loop_start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Resize once (used for sending & overlay)
                frame_resized = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
                
                with self.latest_frame_lock:
                    self.latest_frame = frame_resized
                
                # For display, overlay detections on copy
                display_frame = frame_resized.copy()
                display_frame = self.draw_detections(display_frame)
                
                # Add FPS counter
                fps_text = self.update_fps_display()
                if fps_text:
                    cv2.putText(display_frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add model info
                model_text = f"Model: {self.model_name}"
                cv2.putText(display_frame, model_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Scale back to original size for display (optional)
                display_frame = cv2.resize(display_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                cv2.imshow("Production AI Vision Client", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):  # Change model type on the fly (simplified)
                    # Cycle through model types
                    if self.model_type == "yolo":
                        self.model_type = "mediapipe"
                        self.model_name = "mediapipe_pose"
                    elif self.model_type == "mediapipe":
                        self.model_type = "yolo"
                        self.model_name = "yolo_yolo11m"
                    print(f"Switched to {self.model_type} - {self.model_name}")
                
                elapsed = time.time() - loop_start
                sleep_left = display_interval - elapsed
                if sleep_left > 0:
                    time.sleep(sleep_left)
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.running = False
            time.sleep(0.2)
            if self.cap:
                self.cap.release()
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            print("Client stopped")

    def list_available_models(self) -> Dict[str, Any]:
        """Request list of available models from server"""
        if self.ws and self.ws.connected:
            try:
                # Send command to list models
                self.ws.send(json.dumps({"command": "list_models"}))
                response = self.ws.recv()
                return json.loads(response)
            except Exception as e:
                print(f"Error requesting model list: {e}")
                return {}
        else:
            print("Not connected to server")
            return {}


if __name__ == "__main__":
    # Example usage with different model options
    
    # You can change these parameters based on your needs
    client = ProductionClient(
        server_url="ws://localhost:8000/ws",  # Update this to your server address
        model_type="yolo",  # or "mediapipe"
        model_name="yolo_yolo11m",  # or "mediapipe_pose", "mediapipe_hands", etc.
        capture_index=0,  # Camera index (0 for default, 2 for external USB camera)
        display_fps=30,
        send_fps=10,
        width=640,
        height=480,
        jpeg_quality=60,
        motion_threshold=3.0
    )
    
    # Optionally, you can list available models before starting
    print("Available models:")
    # client.list_available_models()  # Call when connected in real usage
    
    print("Starting client...")
    print("Press 'q' to quit, 'm' to switch between YOLO and MediaPipe models")
    client.run()