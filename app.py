from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from asl_reader import ASL_LETTERS, calculate_finger_angles, predict_letter
import time
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# Security headers
@app.after_request
def add_header(response):
    response.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
    })
    return response

# Initialize MediaPipe Hands once
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def init_camera():
    """Initialize camera with basic settings"""
    for camera_id in range(2):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            # Warm up camera
            for _ in range(3):
                cap.read()
                time.sleep(0.1)
            return cap
    return None

def process_frame(frame, max_dimension=800):
    """Process a single frame with all necessary transformations"""
    if frame is None:
        return None

    # Resize if needed
    height, width = frame.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            angles = calculate_finger_angles(hand_landmarks)
            predicted_letter = predict_letter(angles)

            # Draw angles and predictions
            for i, angle in enumerate(angles):
                cv2.putText(frame, f'Finger {i}: {round(np.degrees(angle), 2)}',
                          (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if predicted_letter:
                cv2.putText(frame, f'Letter: {predicted_letter}',
                          (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return frame

def generate_frames():
    cap = init_camera()
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        try:
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_camera')
def check_camera():
    cap = init_camera()
    status = "ok" if cap and cap.read()[0] else "error"
    if cap:
        cap.release()
    return jsonify({"status": status})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True) 