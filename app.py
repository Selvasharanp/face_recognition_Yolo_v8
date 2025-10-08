from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
from face_utils import FaceRecognitionSystem
import base64
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)
face_system = FaceRecognitionSystem()

# Global variables
camera = None
detection_active = False
detection_history = []

def generate_frames():
    global camera, detection_active
    while True:
        if camera and detection_active:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Process frame for face recognition
                processed_frame, face_info = face_system.recognize_faces(frame)
                
                # Add to history with duplicate prevention
                current_time = datetime.now()
                for info in face_info:
                    # Only add if not a duplicate within last 5 minutes
                    if not is_duplicate_detection(info, detection_history):
                        detection_history.append(info)
                        # Keep only last 50 unique detections
                        if len(detection_history) > 50:
                            detection_history.pop(0)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def is_duplicate_detection(new_detection, history):
    """Check if this person was detected in the last 5 minutes"""
    if new_detection["name"] == "Unknown":
        return False
        
    current_time = datetime.now()
    five_minutes_ago = current_time - timedelta(minutes=5)
    
    for detection in reversed(history):  # Check most recent first
        if detection["name"] == new_detection["name"]:
            detection_time = datetime.strptime(detection["time"], "%Y-%m-%d %H:%M:%S")
            if detection_time > five_minutes_ago:
                return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera, detection_active
    if camera is None:
        camera = cv2.VideoCapture(0)
    detection_active = True
    return jsonify({"status": "camera started"})

@app.route('/stop_camera')
def stop_camera():
    global camera, detection_active
    detection_active = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "camera stopped"})

@app.route('/get_detections')
def get_detections():
    return jsonify(detection_history)

@app.route('/add_face', methods=['POST'])
def add_face():
    data = request.json
    image_data = data['image'].split(',')[1]
    name = data['name']
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Add to known faces
    face_system.add_new_face(image, name)
    
    return jsonify({"status": "face added"})

@app.route('/get_known_people')
def get_known_people():
    people = face_system.get_known_people()
    return jsonify(people)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)