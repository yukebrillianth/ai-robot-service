"""
Integration test for the production AI Vision Server and Client
This script tests basic connectivity and model selection functionality
"""

import asyncio
import json
import threading
import time
from websocket import create_connection

def test_server_connection():
    """Test basic server connection and model selection"""
    print("Testing server connection...")
    
    try:
        # Connect to server
        ws = create_connection("ws://localhost:8000/ws")
        print("✓ Connected to server")
        
        # Send model selection
        model_selection = {
            "model_type": "yolo",
            "model_name": "yolo_yolo11m"
        }
        ws.send(json.dumps(model_selection))
        print("✓ Sent model selection")
        
        # Wait for confirmation
        response = ws.recv()
        result = json.loads(response)
        
        if result.get("status") == "connected":
            print(f"✓ Server confirmed connection with model: {result['model']}")
        else:
            print(f"✗ Server connection failed: {result}")
            return False
        
        # Test model listing
        ws.send(json.dumps({"command": "list_models"}))
        response = ws.recv()
        model_list = json.loads(response)
        
        if "models" in model_list:
            print(f"✓ Server has {len(model_list['models'])} available models")
            print(f"  Available models: {model_list['models']}")
        else:
            print(f"✗ Failed to get model list: {model_list}")
            return False
        
        # Close connection
        ws.close()
        print("✓ Connection closed")
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def test_different_models():
    """Test different model types"""
    models_to_test = [
        ("yolo", "yolo_yolo11m"),
        ("yolo", "yolo_yolov8n"),
        ("mediapipe", "mediapipe_pose"),
        ("mediapipe", "mediapipe_hands")
    ]
    
    print("\nTesting different models...")
    
    for model_type, model_name in models_to_test:
        try:
            ws = create_connection("ws://localhost:8000/ws")
            
            # Send model selection
            model_selection = {
                "model_type": model_type,
                "model_name": model_name
            }
            ws.send(json.dumps(model_selection))
            
            # Wait for confirmation
            response = ws.recv()
            result = json.loads(response)
            
            if result.get("status") == "connected":
                print(f"✓ Successfully connected with {model_type} - {model_name}")
            else:
                print(f"✗ Failed to connect with {model_type} - {model_name}: {result}")
            
            ws.close()
            
        except Exception as e:
            print(f"✗ Error testing {model_type} - {model_name}: {e}")

def run_tests():
    """Run all integration tests"""
    print("Starting AI Vision Server Integration Tests\n")
    
    # Test 1: Basic connection
    if test_server_connection():
        print("\n✓ Basic connection test passed")
    else:
        print("\n✗ Basic connection test failed")
        return False
    
    # Test 2: Different models
    test_different_models()
    
    print("\nIntegration tests completed!")
    return True

if __name__ == "__main__":
    print("This test script requires the server to be running at ws://localhost:8000/ws")
    print("Start the server first using: python server/production_server.py")
    
    run_tests()