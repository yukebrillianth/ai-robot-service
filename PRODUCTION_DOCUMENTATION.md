# Production AI Vision Server & Client System

## Overview

This production system provides real-time object detection and computer vision capabilities using both YOLO and MediaPipe models with GPU acceleration. The system supports multiple clients connecting simultaneously with the ability to select from different AI models.

## Architecture

### Server Components
- **FastAPI WebSocket Server**: Handles concurrent connections from multiple clients
- **Model Manager**: Manages YOLO and MediaPipe models with dynamic loading
- **Connection Manager**: Tracks active client connections and their model preferences
- **Thread Pool**: Handles CPU-intensive operations without blocking the event loop

### Client Components
- **Camera Capture**: Supports various camera inputs with configurable parameters
- **Frame Processing**: JPEG encoding with adaptive quality based on network performance
- **Motion Detection**: Reduces bandwidth by only sending frames with significant changes
- **Model Selection**: Dynamic switching between YOLO and MediaPipe models

## Available Models

### YOLO Models
- `yolo_yolo11m`: YOLOv11 medium model for general object detection
- `yolo_yolov8n`: YOLOv8 nano model for faster inference
- `yolo_yolov8m`: YOLOv8 medium model for balanced performance

### MediaPipe Models
- `mediapipe_pose`: Human pose estimation and landmark detection
- `mediapipe_hands`: Hand landmark detection and tracking
- `mediapipe_objects`: 3D object detection (objectron)
- `mediapipe_face`: Face detection

## Server Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker and Kubernetes cluster
- Models pre-downloaded or available for download

### Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t ai-vision-server:latest .
   ```

2. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

3. **Verify deployment**:
   ```bash
   kubectl get pods
   kubectl logs -f deployment/ai-vision-server
   ```

## Client Usage

### Basic Client
```python
from client.production_client import ProductionClient

client = ProductionClient(
    server_url="ws://your-server-ip:8000/ws",
    model_type="yolo",  # or "mediapipe"
    model_name="yolo_yolo11m",  # or other available models
    capture_index=0,  # Camera index
    display_fps=30,   # Display frames per second
    send_fps=10,      # Send frames per second
    width=640,        # Frame width
    height=480,       # Frame height
    jpeg_quality=60   # JPEG compression quality (1-100)
)

client.run()
```

### Model Selection
Clients can select their preferred model when connecting:
- The server supports dynamic model switching
- Each client can use a different model simultaneously
- Models are loaded once and shared across clients for the same model type

### Keyboard Controls
- `q`: Quit the application
- `m`: Switch between YOLO and MediaPipe models (cycles through options)

## Client Configuration Options

### Core Configuration
- `server_url`: WebSocket URL of the server (e.g., "ws://localhost:8000/ws")
- `model_type`: Either "yolo" or "mediapipe"
- `model_name`: Specific model identifier (see available models above)
- `capture_index`: Camera device index (0 for default, 2 for external USB camera)

### Performance Configuration
- `display_fps`: Frames per second for display (default: 30)
- `send_fps`: Frames per second to send to server (default: 10)
- `width`, `height`: Frame dimensions (default: 640x480)
- `jpeg_quality`: JPEG compression level (1-100, default: 60)
- `motion_threshold`: Percentage of pixel change required to send frame (default: 4.0%)

### Adaptive Quality Settings
- `min_quality`: Minimum JPEG quality (default: 20)
- `max_quality`: Maximum JPEG quality (default: 80)
- Quality automatically adjusts based on network performance

## Server API Endpoints

### WebSocket Endpoint
- **Path**: `/ws`
- **Protocol**: WebSocket with binary/text payload support
- **Initialization**: Send JSON with `model_type` and `model_name` to select model

### REST Endpoints
- `GET /health`: Health check returning model status and connection count
- `GET /models`: List of available models

### Payload Format
The server supports both binary and text payloads:

#### Binary Payload Format
```
[4-byte header length][JSON header][JPEG data]
```
Header contains metadata like frame dimensions, quality, and timing information.

#### Text Payload Format
Base64-encoded JPEG data for fallback compatibility.

## Detection Response Format

### YOLO Detection
```json
[
  {
    "label": "person",
    "confidence": 0.95,
    "x": 100,
    "y": 150,
    "w": 200,
    "h": 300,
    "type": "object_detection",
    "start_time": 1699123456.789
  }
]
```

### MediaPipe Detection
- **Pose Landmarks**: `{"label": "pose_keypoint_X", "confidence": 0.8, ...}`
- **Hand Landmarks**: `{"label": "hand_X_keypoint_Y", "confidence": 0.8, ...}`
- **Face Detection**: `{"label": "face", "confidence": 0.85, ...}`
- **Object Detection**: `{"label": "object", "confidence": 0.7, ...}`

## Kubernetes Configuration

### GPU Requirements
- Configured for NVIDIA H200 GPUs
- Requests 1 GPU per server instance
- Can be scaled horizontally based on demand

### Resource Limits
- CPU: 8 cores limit, 2 cores request
- Memory: 16GB limit, 4GB request
- GPU: 1 NVIDIA GPU

### Persistent Storage
- PVC for model storage (10GB)
- Models are shared across server replicas

## Performance Considerations

### Server Performance
- Uses asyncio for handling multiple connections efficiently
- GPU-accelerated inference for YOLO models
- Optimized MediaPipe CPU inference
- Thread pool for CPU-intensive operations

### Network Optimization
- Motion detection reduces unnecessary frame transmission
- Adaptive JPEG quality for bandwidth optimization
- Binary payloads reduce overhead compared to base64

### Client Performance
- Separate threads for sending and receiving
- Frame rate control to prevent overwhelming the server
- Local display does not block network operations

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Connection Refused**: Verify server IP and port are accessible
3. **Poor Performance**: Increase GPU resources or reduce frame rate
4. **Model Loading Error**: Ensure model files are available

### Monitoring
- Check server logs: `kubectl logs deployment/ai-vision-server`
- Monitor GPU usage: `kubectl exec deployment/ai-vision-server -- nvidia-smi`
- Check health endpoint: `curl http://your-server/health`

## Scaling

### Horizontal Scaling
- Deploy multiple server instances behind a load balancer
- Each client maintains its own WebSocket connection
- Consider sticky sessions for consistent model selection

### Load Testing
- Monitor active connection count via health endpoint
- Track inference latency per client
- Watch GPU utilization metrics

## Security Considerations

### Network Security
- Use TLS for production deployments
- Implement authentication if needed
- Validate model selection parameters

### Container Security
- Runs as non-root user
- Minimal required permissions
- Regular security updates

## Development and Customization

### Adding New Models
1. Add model loading to `ModelManager.load_default_models()`
2. Implement detection conversion in `convert_*_results_to_detections()`
3. Update type definitions for new model types

### Client Customization
- Modify drawing functions for different visualization styles
- Adjust motion detection sensitivity
- Add new keyboard controls

## Production Deployment Checklist

- [ ] Verify GPU resources are available in Kubernetes
- [ ] Test with expected number of concurrent clients
- [ ] Set up monitoring for performance metrics
- [ ] Configure health checks and auto-scaling
- [ ] Implement backup strategy for model data
- [ ] Set up logging aggregation
- [ ] Review security configurations
- [ ] Plan for graceful updates

## Example Use Cases

### Object Detection
```python
client = ProductionClient(
    model_type="yolo",
    model_name="yolo_yolo11m",
    # Detects people, vehicles, animals, etc.
)
```

### Human Pose Estimation
```python
client = ProductionClient(
    model_type="mediapipe",
    model_name="mediapipe_pose",
    # Tracks human body landmarks
)
```

### Hand Tracking
```python
client = ProductionClient(
    model_type="mediapipe",
    model_name="mediapipe_hands",
    # Tracks hand pose and gestures
)
```

This system is designed for production use with multiple clients, GPU acceleration, and robust error handling.