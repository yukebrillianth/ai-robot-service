import base64
import json
import threading
import time
from collections import deque

import cv2
from websocket import WebSocket


class CameraClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.ws = None
        self.cap = None
        self.detections = deque(maxlen=1)
        self.lock = threading.Lock()
        
    def connect(self):
        """Connect to WebSocket server"""
        try:
            self.ws = WebSocket()
            self.ws.connect(self.server_url)
            print("Connected to server")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            print("Failed to open camera")
            return False
        return True
    
    def encode_frame(self, frame):
        """Encode frame to base64 JPEG"""
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    
    def receive_detections(self):
        """Continuously receive detection results"""
        while True:
            try:
                if self.ws and self.ws.connected:
                    result = self.ws.recv()
                    detections = json.loads(result)
                    if detections and isinstance(detections, list) and 'start_time' in detections[0]:
                        now = time.time()
                        latency_ms = (now - detections[0]['start_time']) * 1000.0
                        print(f"Latency: {latency_ms:.2f} ms")
                    with self.lock:
                        self.detections.append(detections)
            except Exception as e:
                print(f"Error receiving detections: {e}")
                break
    
    def draw_detections(self, frame):
        """Draw bounding boxes on frame"""
        with self.lock:
            if len(self.detections) > 0:
                detections = self.detections[-1]  # Get latest detections
                for detection in detections:
                    x = detection['x']
                    y = detection['y']
                    w = detection['w']
                    h = detection['h']
                    label = detection['label']
                    confidence = detection['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame
    
    def run(self):
        """Main loop"""
        if not self.connect():
            return
            
        if not self.start_camera():
            return
            
        # Start receiving detections in separate thread
        receiver_thread = threading.Thread(target=self.receive_detections)
        receiver_thread.daemon = True
        receiver_thread.start()
        
        # Frame rate control
        fps = 10
        delay = 1.0 / fps
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Resize frame for faster transmission
                frame_resized = cv2.resize(frame, (640, 360))
                
                # Encode and send frame
                encoded_frame = self.encode_frame(frame_resized)
                try:
                    self.ws.send(encoded_frame)
                except Exception as e:
                    print(f"Failed to send frame: {e}")
                    break
                
                # Draw detections on original frame
                frame_with_detections = self.draw_detections(frame_resized)
                frame_with_detections = cv2.resize(frame_with_detections, (frame.shape[1], frame.shape[0]))
                # Display frame
                cv2.imshow('Robot AI Vision', frame_with_detections)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Control frame rate
                elapsed = time.time() - start_time
                sleep_time = delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            if self.cap:
                self.cap.release()
            if self.ws:
                self.ws.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    client = CameraClient("ws://ai-server.yukebrillianth.my.id/ws")
    client.run()