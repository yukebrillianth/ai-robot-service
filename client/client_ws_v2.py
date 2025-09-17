import base64
import json
import threading
import time
from collections import deque

import cv2
from websocket import WebSocket


class CameraClient:
    def __init__(
        self,
        server_url,
        capture_index=2,
        display_fps=30,
        send_fps=10,
        width=640,
        height=480,
        jpeg_quality=60,
        motion_threshold=4.0  # persen perbedaan piksel untuk kirim
    ):
        self.server_url = server_url
        self.capture_index = capture_index
        self.display_fps = display_fps
        self.send_fps = send_fps
        self.resize_w = width
        self.resize_h = height
        self.jpeg_quality = jpeg_quality
        self.motion_threshold = motion_threshold

        self.ws = None
        self.cap = None

        self.detections = deque(maxlen=1)
        self.detections_lock = threading.Lock()

        self.latest_frame = None          # frame terakhir (resized) untuk dikirim
        self.latest_frame_lock = threading.Lock()

        self.prev_sent_gray = None        # untuk motion detection sederhana

        self.running = False

    def connect(self):
        try:
            self.ws = WebSocket()
            self.ws.settimeout(2)  # supaya recv tidak block terlalu lama
            self.ws.connect(self.server_url)
            print("Connected to server")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.capture_index)
        if not self.cap.isOpened():
            print("Failed to open camera")
            return False
        return True

    def encode_frame(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ok:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    def receive_detections(self):
        while self.running:
            try:
                if self.ws and self.ws.connected:
                    try:
                        result = self.ws.recv()
                    except Exception:
                        continue  # timeout / socket issue -> loop lagi
                    detections = json.loads(result)
                    if (detections and isinstance(detections, list)
                            and 'start_time' in detections[0]):
                        now = time.time()
                        latency_ms = (now - detections[0]['start_time']) * 1000.0
                        print(f"Latency: {latency_ms:.2f} ms")
                    with self.detections_lock:
                        self.detections.append(detections)
            except Exception as e:
                if self.running:
                    print(f"Error receiving detections: {e}")
                time.sleep(0.5)

    def draw_detections(self, frame):
        with self.detections_lock:
            if len(self.detections) == 0:
                return frame
            detections = self.detections[-1]
        for det in detections:
            try:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                label = det.get('label', '')
                conf = det.get('confidence', 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{label}:{conf:.2f}",
                            (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1)
            except KeyError:
                continue
        return frame

    def frame_should_send(self, gray_frame):
        # Pertama kali pasti kirim
        if self.prev_sent_gray is None:
            self.prev_sent_gray = gray_frame
            return True
        # Hitung perbedaan rata-rata
        diff = cv2.absdiff(self.prev_sent_gray, gray_frame)
        mean_diff = diff.mean()
        # Normalisasi kasar: jika perbedaan intensitas > threshold persen (skala 0-255)
        percent = (mean_diff / 255.0) * 100.0
        if percent >= self.motion_threshold:
            self.prev_sent_gray = gray_frame
            return True
        return False

    def sender_loop(self):
        send_interval = 1.0 / max(1, self.send_fps)
        while self.running:
            start = time.time()
            frame = None
            with self.latest_frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is not None and self.ws and self.ws.connected:
                # Motion check
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self.frame_should_send(gray):
                    encoded = self.encode_frame(frame)
                    if encoded:
                        try:
                            self.ws.send(encoded)
                        except Exception as e:
                            print(f"Failed to send frame: {e}")
            # Sleep sisa waktu
            elapsed = time.time() - start
            to_sleep = send_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def run(self):
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

                # Resize sekali (dipakai untuk kirim & display)
                frame_resized = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)

                # Update latest frame tanpa blocking lama
                with self.latest_frame_lock:
                    self.latest_frame = frame_resized

                # Copy untuk display + overlay (tidak ganggu yang akan dikirim)
                display_frame = frame_resized.copy()
                display_frame = self.draw_detections(display_frame)
                display_frame = cv2.resize(display_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

                cv2.imshow("Robot AI Vision", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Jaga FPS display
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


if __name__ == "__main__":
    client = CameraClient(
        "ws://ai-server.yukebrillianth.my.id/ws",
        capture_index=2,
        display_fps=30,
        send_fps=10,
        width=640,
        height=360,
        jpeg_quality=20,
        motion_threshold=3.0
    )
    client.run()