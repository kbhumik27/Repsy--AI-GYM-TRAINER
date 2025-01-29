from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import tensorflow as tf
import intel_extension_for_tensorflow as itex  # Import ITEX for optimization

# Enable ITEX optimizations
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to map all joints
def map_joints():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        results = pose.process(image)

        # Convert image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if landmarks are detected
        if results.pose_landmarks:
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

        # Encode image to JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        # Yield frame to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(map_joints(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
