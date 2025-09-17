import asyncio
import base64
import json
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from ultralytics import YOLO

app = FastAPI()

# Load YOLOv8n model
model = YOLO('yolo11m.pt')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            # Decode base64 image
            image_data = base64.b64decode(data)
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Frame rate control
            fps = 30
            delay = 1.0 / fps

            start_time = time.time()
                    
            cv2.imshow('Robot AI Vision', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Control frame rate
            # elapsed = time.time() - start_time
            # sleep_time = delay - elapsed
            # if sleep_time > 0:
            #     time.sleep(sleep_time)

            print(f"Received frame size: {frame.shape if frame is not None else 'None'}")
            
            if frame is None:
                continue
                
            # Run YOLOv8 inference
            results = model(frame, device='mps')
            
            # Extract detection results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                        print(f"Detected box: xyxy={xyxy} x={x1}, y={y1}, w={w}, h={h}")
                        
                        # Get class and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        label = model.names[cls]
                        
                        detections.append({
                            "label": label,
                            "confidence": conf,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h
                        })
            
            # Send results back to client
            await websocket.send_text(json.dumps(detections))
            
    except Exception as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        await websocket.close()
        print("Client disconnected")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)