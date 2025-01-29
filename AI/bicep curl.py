from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex  # Import ITEX for optimization

# Enable ITEX optimizations
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Global variables for exercise counters
exercise_counters = {
    'bicep_curl': 0
}

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle 

# Function to check bicep curl form
def check_bicep_curl_form(shoulder, elbow, wrist):
    angle = calculate_angle(shoulder, elbow, wrist)
    if angle > 160:  # Arm is raised
        return "Raise your arm fully"
    elif angle < 90:  # Arm is not curled enough
        return "Curl your arm more"
    else:
        return "Great form!"

# Main function for bicep curl exercise
def bicep_curl():
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        results = pose.process(image)

        # Convert image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for bicep curl exercise
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0]]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]]

            # Render form feedback
            form_feedback = check_bicep_curl_form(shoulder, elbow, wrist)
            cv2.putText(image, form_feedback, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 2, cv2.LINE_AA)

            # Check if the form is "Great form!"
            if form_feedback == "Great form!":
                exercise_counters['bicep_curl'] += 1
                cv2.putText(image, f"Rep Count: {exercise_counters['bicep_curl']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(e)

        # Encode image to JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        # Yield frame to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Main route to render the main website
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(bicep_curl(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', port=5004, debug=True)
