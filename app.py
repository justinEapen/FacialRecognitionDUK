from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import threading
from queue import Queue
import time
import traceback

app = Flask(__name__)
CORS(app)

# Enable Flask debug mode
app.config['DEBUG'] = True

# Create necessary directories
Path("data/faces").mkdir(parents=True, exist_ok=True)
Path("data/logs").mkdir(parents=True, exist_ok=True)
Path("data/temp").mkdir(parents=True, exist_ok=True)

# Initialize global variables
known_face_encodings = []
known_face_names = []
face_data = {}
camera = None
frame_queue = Queue(maxsize=10)
stop_camera = False

def load_registered_faces():
    """Load all registered faces from the data directory."""
    global known_face_encodings, known_face_names, face_data
    
    known_face_encodings = []
    known_face_names = []
    
    try:
        if os.path.exists('data/face_data.json'):
            with open('data/face_data.json', 'r') as f:
                face_data = json.load(f)
                
            for person_id, data in face_data.items():
                if os.path.exists(f"data/faces/{person_id}.npy"):
                    encoding = np.load(f"data/faces/{person_id}.npy")
                    known_face_encodings.append(encoding)
                    known_face_names.append(data['name'])
    except Exception as e:
        print(f"Error loading faces: {str(e)}")

def log_access_attempt(person_id, status):
    """Log access attempts with timestamp."""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'person_id': person_id,
        'status': status
    }
    
    log_file = 'data/logs/access_log.csv'
    pd.DataFrame([log_entry]).to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/register', methods=['POST'])
def register_face():
    """Register a new face."""
    try:
        print("Received registration request")
        print("Files:", request.files)
        print("Form data:", request.form)
        
        if 'image' not in request.files:
            print("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        name = request.form.get('name')
        person_id = request.form.get('id')
        
        print(f"Processing registration for {name} (ID: {person_id})")
        
        if not name or not person_id:
            print("Missing name or ID")
            return jsonify({'error': 'Name and ID are required'}), 400
        
        # Create a temporary file to save the uploaded image
        temp_path = os.path.join('data', 'temp', f'{person_id}_temp.jpg')
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        try:
            # Save and process the image
            print(f"Saving image to {temp_path}")
            image_file.save(temp_path)
            
            print("Loading image file")
            image = face_recognition.load_image_file(temp_path)
            
            print("Detecting faces")
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                print("No faces detected")
                os.remove(temp_path)
                return jsonify({'error': 'No face detected in the image'}), 400
                
            if len(face_locations) > 1:
                print("Multiple faces detected")
                os.remove(temp_path)
                return jsonify({'error': 'Multiple faces detected. Please upload an image with only one face'}), 400
                
            print("Computing face encoding")
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            print("Saving face encoding")
            np.save(f"data/faces/{person_id}.npy", face_encoding)
            
            print("Updating face data")
            face_data[person_id] = {
                'name': name,
                'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('data/face_data.json', 'w') as f:
                json.dump(face_data, f)
                
            print("Reloading registered faces")
            load_registered_faces()
            
            print("Registration successful")
            return jsonify({'success': True, 'message': 'Face registered successfully'})
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print("Error in register_face:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def camera_stream():
    """Background thread for capturing camera frames."""
    global camera, stop_camera
    
    while not stop_camera:
        if camera is None or not camera.isOpened():
            try:
                if camera is not None:
                    camera.release()
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    print("Failed to open camera")
                    time.sleep(1)
                    continue
            except Exception as e:
                print(f"Error initializing camera: {str(e)}")
                time.sleep(1)
                continue

        try:
            success, frame = camera.read()
            if not success:
                print("Failed to read frame")
                camera.release()
                camera = None
                continue

            # Clear queue if full
            while frame_queue.full():
                frame_queue.get()
            
            frame_queue.put(frame)
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
            
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            if camera is not None:
                camera.release()
                camera = None
            time.sleep(1)

def gen_frames():
    """Generate camera frames with face recognition."""
    while True:
        if frame_queue.empty():
            time.sleep(0.01)  # Small delay if no frames
            continue
            
        try:
            frame = frame_queue.get()
            
            # Reduce frame size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    log_access_attempt(name, "Authorized")
                else:
                    log_access_attempt("Unknown", "Unauthorized")
                    
                face_names.append(name)
                
            # Draw results on frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_logs')
def get_logs():
    """Retrieve access logs."""
    try:
        logs = pd.read_csv('data/logs/access_log.csv')
        return jsonify(logs.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start camera thread
    camera_thread = threading.Thread(target=camera_stream, daemon=True)
    camera_thread.start()
    
    try:
        load_registered_faces()
        app.run(debug=True, use_reloader=False)
    finally:
        stop_camera = True
        if camera is not None:
            camera.release()
