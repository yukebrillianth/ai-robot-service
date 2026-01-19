import json
import struct
import threading
import time
from collections import deque

import cv2
from websocket import ABNF, WebSocket


class CameraClient:
    def __init__(
        self,
        server_url,
        capture_index=0,
        display_fps=30,
        send_fps=10,
        width=320,
        height=240,
        jpeg_quality=50,
        motion_threshold=4.0,   # persen perbedaan piksel untuk kirim
        min_quality=20,
        max_quality=80,
        motion_downscale=(160, 120),  # ukuran untuk kalkulasi motion (lebih kecil => lebih cepat)
    ):
        self.server_url = server_url
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
        self.latest_frame = None          # frame terakhir (resized) untuk dikirim (BGR)
        self.latest_frame_lock = threading.Lock()
        self.prev_sent_gray_small = None  # untuk motion detection di ukuran kecil
        self.running = False

        # stats untuk adaptasi kualitas
        self.send_time_ema = None
        self.ema_alpha = 0.2

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
        # Untuk Raspi, pertimbangkan pakai libcamera / PiCamera untuk performa lebih baik
        self.cap = cv2.VideoCapture(self.capture_index)
        if not self.cap.isOpened():
            print("Failed to open camera")
            return False
        # kecilkan buffer capture pada beberapa driver jika perlu:
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return True

    def encode_frame_bytes(self, frame_bgr, quality):
        # frame_bgr: BGR resized sesuai self.resize_w/self.resize_h
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
        if not ok:
            return None
        return buffer.tobytes()

    def _build_payload(self, jpeg_bytes):
        # Format: 4-byte BE header_length + header_json + jpeg_bytes
        meta = {
            "width": self.resize_w,
            "height": self.resize_h,
            "format": "jpeg",
            "jpeg_quality": int(self.jpeg_quality),
            "start_time": time.time()
        }
        header = json.dumps(meta).encode('utf-8')
        prefix = struct.pack('>I', len(header))
        return prefix + header + jpeg_bytes

    def receive_detections(self):
        while self.running:
            try:
                if self.ws and self.ws.connected:
                    try:
                        result = self.ws.recv()
                    except Exception:
                        continue  # timeout / socket issue
                    # Ditempatkan asumsinya server mengirim JSON teks untuk detections
                    try:
                        detections = json.loads(result)
                        if (detections and isinstance(detections, list)
                                and 'start_time' in detections[0]):
                            now = time.time()
                            round_trip_ms = (now - detections[0]['start_time']) * 1000.0
                            latency_ms = (round_trip_ms - detections[0]['process_duration_ms'])
                            print(f"Round Trip: {round_trip_ms:.2f} ms")
                            print(f"Process Time: {detections[0]['process_duration_ms']} ms")
                            print(f"Latency: {latency_ms:.2f} ms")
                        with self.detections_lock:
                            self.detections.append(detections)
                    except Exception as e:
                        # jika bukan JSON, abaikan atau tampilkan debug
                        # print("Non-JSON message:", e)
                        continue
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
                x, y, w, h = map(int, (det['x'], det['y'], det['w'], det['h']))
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

    def frame_should_send(self, frame_bgr):
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

    def adjust_quality_based_on_send(self, send_time):
        # EMA untuk waktu kirim
        if self.send_time_ema is None:
            self.send_time_ema = send_time
        else:
            self.send_time_ema = (1 - self.ema_alpha) * self.send_time_ema + self.ema_alpha * send_time
        # jika pengiriman lambat (lebih dari interval send), turunkan quality sedikit
        target = 1.0 / max(1, self.send_fps)
        if self.send_time_ema > target * 0.8:
            self.jpeg_quality = max(self.min_quality, int(self.jpeg_quality * 0.9))
        else:
            # kalau masih cepat, naikkan sedikit ke limit
            self.jpeg_quality = min(self.max_quality, int(self.jpeg_quality * 1.05))

    def sender_loop(self):
        send_interval = 1.0 / max(1, self.send_fps)
        while self.running:
            start = time.time()
            frame = None
            with self.latest_frame_lock:
                if self.latest_frame is not None:
                    # gunakan reference copy minimal
                    frame = self.latest_frame.copy()
            if frame is not None and self.ws and self.ws.connected:
                if self.frame_should_send(frame):
                    # encode ke jpeg bytes (langsung)
                    jpeg_bytes = self.encode_frame_bytes(frame, self.jpeg_quality)
                    if jpeg_bytes:
                        payload = self._build_payload(jpeg_bytes)
                        try:
                            send_start = time.time()
                            # kirim sebagai binary
                            self.ws.send(payload, opcode=ABNF.OPCODE_BINARY)
                            send_time = time.time() - send_start
                            # adaptif quality
                            self.adjust_quality_based_on_send(send_time)
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
                # Resize sekali (dipakai untuk kirim & overlay)
                frame_resized = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
                with self.latest_frame_lock:
                    self.latest_frame = frame_resized
                # Untuk display, overlay detections pada salinan
                display_frame = frame_resized.copy()
                display_frame = self.draw_detections(display_frame)
                # kembalikan ke ukuran asli untuk display (opsional)
                display_frame = cv2.resize(display_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Robot AI Vision", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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
        "ws://ai-robot-route-robot-ai-its.apps.iohairan.nokia-airan.ioh.com/ws",
        capture_index=2,
        display_fps=30,
        send_fps=30,
        width=640,
        height=360,
        jpeg_quality=20,
        motion_threshold=3.0
    )
    client.run()