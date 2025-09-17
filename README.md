# AI Robot Vision with Cloud Inference (WebSocket Only)

This project implements a real-time object detection system where a robot (Raspberry Pi) captures camera frames, sends them to a cloud server via WebSocket, and receives object detection results back for visualization.

## Project Structure
```
robot-ai-server/
├── server/
│   └── main.py          # FastAPI WebSocket server with YOLOv8
├── client/
│   └── client_ws.py     # Raspberry Pi client with camera capture
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup Environment

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv8n model (will be automatically downloaded on first run):
   The model will be automatically downloaded when you first run the server.

## Running the Server

1. Navigate to the project directory:
   ```bash
   cd robot-ai-server
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run the FastAPI server:
   ```bash
   python server/main.py
   ```

   The server will start on `http://localhost:8000`

## Running the Client (Raspberry Pi)

1. Navigate to the project directory:
   ```bash
   cd robot-ai-server
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Run the client:
   ```bash
   python client/client_ws.py
   ```

   The client will connect to the server at `ws://localhost:8000/ws` and start capturing camera frames.

## Usage

1. Start the server first
2. Start the client (on the same machine or another machine)
3. The client will open a window showing the camera feed with object detection bounding boxes
4. Press 'q' to quit the application

## Notes

- The system runs at approximately 10 FPS to balance performance and network load
- Frame size is reduced to 320x240 for faster transmission
- YOLOv8n model provides a good balance between accuracy and speed
- The client and server can run on different machines by changing the server URL in the client code