# Robot AI Vision Project - QWEN Context

## Project Overview

This is a real-time object detection system that enables a robot (specifically targeting Raspberry Pi) to capture camera frames, send them to a cloud server via WebSocket, and receive object detection results back for visualization. The system uses YOLO models for AI-powered object detection and is designed for efficient streaming with frame rate control and motion-based frame sending.

### Key Technologies
- **Python 3** - Main programming language
- **FastAPI** - Modern, fast web framework for the server API
- **Uvicorn** - ASGI server for running the application
- **OpenCV** - Computer vision library for image processing
- **YOLO (You Only Look Once)** - Real-time object detection models
- **WebSockets** - Real-time bidirectional communication between client and server
- **NumPy** - Numerical computing library
- **PyTorch** - Machine learning framework
- **Ultralytics** - YOLO model implementation

### Architecture
- **Server**: Running on cloud infrastructure, receives image frames, performs YOLO inference, and sends back detection results
- **Client**: Running on robot device (intended for Raspberry Pi), captures camera feed, sends frames to server, receives detections, and visualizes results

## Project Structure

```
robot-ai-server/
├── server/
│   ├── main.py          # Primary WebSocket server with YOLO inference
│   └── server2.py       # Alternative WebSocket server implementation (handles binary payloads)
├── client/
│   ├── client_ws.py     # Primary client implementation
│   └── client_ws_v2.py  # Alternative client implementation (simpler binary format)
├── requirements.txt     # Python dependencies
├── run_server.sh        # Script to start the server
├── run_client.sh        # Script to start the client
├── setup.sh             # Script to set up the environment
├── check_deps.py        # Script to check dependencies
├── README.md            # Project documentation
├── yolo11m.pt           # YOLOv11 medium model file
├── yolov5nu.pt          # YOLOv5 nano model file
├── yolov8m.pt           # YOLOv8 medium model file
├── yolov8n.pt           # YOLOv8 nano model file
└── venv/                # Python virtual environment (gitignored)
```

## Building and Running

### Environment Setup
1. Create and activate a virtual environment:
```bash
source setup.sh
# OR manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Server
1. Start the server:
```bash
python server/main.py
# OR using the script:
./run_server.sh
```
The server will start on `http://localhost:8000` (or `0.0.0.0:8000` to accept external connections).

### Running the Client
1. Start the client:
```bash
python client/client_ws.py
# OR using the script:
./run_client.sh
```
The client will connect to the server and start capturing camera frames.

## Key Features

### Server Features
- WebSocket endpoint `/ws` for receiving frames and sending detections
- YOLO model inference with configurable model selection
- Real-time frame display on the server side
- Support for both text (base64) and binary (structured) payload formats
- Motion detection capabilities
- Frame rate control (currently set to 30 FPS)

### Client Features
- Camera capture from various sources (configurable capture index)
- Frame encoding and compression (JPEG)
- Motion-based frame sending to reduce unnecessary network traffic
- Adaptive JPEG quality based on network performance
- Frame rate control for display and sending
- Real-time visualization with bounding boxes
- Thread-based architecture for concurrent sending and receiving

### Communication Protocol
The system supports two payload formats:
1. **Text-based**: Base64 encoded JPEG images
2. **Binary-based**: Structured format with 4-byte header length + JSON header + JPEG bytes

The binary format is more efficient and includes metadata like frame dimensions, JPEG quality, and timing information.

### Performance Optimizations
- Frame size reduction for faster transmission
- Configurable frame rates (30 FPS for display, 10 FPS for sending by default)
- Motion detection to avoid sending frames with minimal changes
- Adaptive JPEG quality based on network performance
- Multithreaded client architecture for non-blocking operations

## Configuration Options

### Client Configuration
The client has various configurable parameters:
- `capture_index`: Camera index to use (default: 0 for laptop webcam, 2 for external camera)
- `display_fps`: Frames per second for display (default: 30)
- `send_fps`: Frames per second to send to server (default: 10)
- `width`, `height`: Frame dimensions (default: 640x360)
- `jpeg_quality`: JPEG compression quality (default: 20-60, lower = more compression)
- `motion_threshold`: Percentage of pixel changes to trigger sending (default: 3-4%)

### Server Configuration
The server supports:
- Configurable YOLO model (yolo11m.pt, yolov5nu.pt, yolov8m.pt, yolov8n.pt)
- Device selection for inference (CPU, GPU, or MPS for Mac)
- WebSocket endpoint configuration

## Development Conventions

### Code Style
- Standard Python PEP 8 style formatting
- Clear, descriptive variable and function names
- Inline comments for complex logic
- Class-based approach for both client and server implementations

### Testing
- The project uses end-to-end testing by running client-server communication
- Frame rate and latency monitoring built into the system
- Error handling for connection issues and inference failures

### Model Management
- Multiple YOLO models are pre-downloaded in the project directory
- Models are loaded at server startup
- Model selection is configurable in the server code

## Troubleshooting

### Common Issues
- Camera not found: Check `capture_index` parameter and ensure camera permissions
- Connection issues: Verify server is running and network connectivity
- Performance issues: Adjust frame rate and quality settings based on network capacity
- High latency: Consider running server closer to client geographically

### Dependency Issues
- Run `python check_deps.py` to verify all dependencies
- Ensure virtual environment is activated
- Update pip and reinstall requirements if needed

## Notes
- YOLOv11m model is currently loaded in main.py (not the YOLOv8n as mentioned in README)
- The system is designed for real-time performance balancing accuracy and speed
- Frame rate control helps balance performance and network load
- Both client and server can run on different machines by changing connection URLs
- The project supports both binary and text-based WebSocket communication modes