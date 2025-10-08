import cv2
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import face_recognition
from ultralytics import YOLO

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.recent_detections = {}  # Store recent detections to avoid duplicates
        
        # Load YOLOv8 model for face detection
        try:
            self.face_model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully!")
        except:
            print("Using default YOLOv8 model")
            self.face_model = YOLO('yolov8n.pt')
        
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from known_faces folder with subdirectories"""
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created {known_faces_dir} directory")
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Clear recent detections when reloading faces
        self.recent_detections = {}
        
        # Scan through all subdirectories
        for person_folder in os.listdir(known_faces_dir):
            person_path = os.path.join(known_faces_dir, person_folder)
            
            if os.path.isdir(person_path):
                print(f"Loading faces for: {person_folder}")
                
                # Load all images in this person's folder
                face_count = 0
                for filename in os.listdir(person_path):
                    if filename.endswith(('.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(person_path, filename)
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            encoding = encodings[0]
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(person_folder)
                            face_count += 1
                
                print(f"  Total faces for {person_folder}: {face_count}")
        
        print(f"\n✅ Loaded {len(self.known_face_names)} total face encodings for {len(set(self.known_face_names))} people")
    
    def is_same_person_recently_detected(self, name, current_time):
        """Check if the same person was detected in the last 5 minutes"""
        if name == "Unknown":
            return False
            
        if name in self.recent_detections:
            last_detection_time = self.recent_detections[name]
            time_diff = current_time - last_detection_time
            # If detected within last 5 minutes, return True
            if time_diff.total_seconds() < 300:  # 300 seconds = 5 minutes
                return True
        
        # Update the detection time
        self.recent_detections[name] = current_time
        return False
    
    def add_new_face(self, image, name, folder_name=None):
        """Add a new face to known faces in a specific folder"""
        if folder_name is None:
            folder_name = name
        
        person_folder = os.path.join("known_faces", folder_name)
        
        # Create folder if it doesn't exist
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
            print(f"Created new folder: {person_folder}")
        
        # Count existing images to create unique filename
        existing_images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        next_number = len(existing_images) + 1
        
        filename = f"{folder_name}_{next_number}.jpg"
        filepath = os.path.join(person_folder, filename)
        
        # Save the image
        cv2.imwrite(filepath, image)
        print(f"Saved new face: {filepath}")
        
        # Reload known faces
        self.load_known_faces()
    
    def detect_faces_yolov8(self, frame):
        """Detect faces using YOLOv8"""
        # Run YOLOv8 inference
        results = self.face_model(frame, verbose=False)
        
        face_locations = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Filter for high confidence detections
                    if conf > 0.6:  # Increased threshold for better accuracy
                        # Convert to face_recognition format (top, right, bottom, left)
                        face_locations.append((int(y1), int(x2), int(y2), int(x1)))
        
        return face_locations
    
    def recognize_faces(self, frame):
        """Recognize faces in the frame using YOLOv8 for detection and face_recognition for recognition"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using YOLOv8
        face_locations = self.detect_faces_yolov8(rgb_frame)
        
        # If YOLOv8 doesn't detect faces, fall back to face_recognition's detector
        if not face_locations:
            face_locations = face_recognition.face_locations(rgb_frame)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_info = []
        current_time = datetime.now()
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "0%"
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                # Convert distance to confidence (0-100%)
                confidence_score = max(0, 100 - (best_distance * 100))
                confidence = f"{confidence_score:.1f}%"
                
                # Use stricter threshold for recognition
                if matches[best_match_index] and best_distance < 0.5:  # Stricter threshold
                    name = self.known_face_names[best_match_index]
                    
                    # Only add to detection list if not detected in last 5 minutes
                    if self.is_same_person_recently_detected(name, current_time):
                        continue  # Skip this detection
            
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with background
            label_background_height = 50 if name != "Unknown" else 35
            cv2.rectangle(frame, (left, bottom - label_background_height), (right, bottom), color, cv2.FILLED)
            
            # Add name text
            text_color = (255, 255, 255)
            cv2.putText(frame, name, (left + 6, bottom - 30 if name != "Unknown" else bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)
            
            # Add confidence for known faces
            if name != "Unknown":
                cv2.putText(frame, f"Conf: {confidence}", (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Add YOLOv8 badge
            cv2.putText(frame, "YOLOv8", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            face_info.append({
                "name": name,
                "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "location": [int(left), int(top), int(right), int(bottom)],
                "confidence": confidence,
                "detector": "YOLOv8" if len(face_locations) > 0 else "Face_Recognition"
            })
        
        return frame, face_info
    
    def get_known_people(self):
        """Get list of all known people (folder names)"""
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            return []
        
        people = []
        for item in os.listdir(known_faces_dir):
            item_path = os.path.join(known_faces_dir, item)
            if os.path.isdir(item_path):
                people.append(item)
        
        return sorted(people)