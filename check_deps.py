import asyncio
import base64
import cv2
import numpy as np
import json
import threading
import time
import websocket
from collections import deque

# Server dependencies
try:
    from fastapi import FastAPI, WebSocket
    import uvicorn
    from ultralytics import YOLO
    print("✓ FastAPI, Uvicorn, and Ultralytics imports successful")
except ImportError as e:
    print(f"✗ Server dependency import error: {e}")

# Client dependencies
try:
    import cv2
    import base64
    import websocket
    import json
    import time
    import threading
    from collections import deque
    print("✓ Client dependencies imports successful")
except ImportError as e:
    print(f"✗ Client dependency import error: {e}")

print("Dependency check complete")