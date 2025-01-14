import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle 

# Function to check dumbbell rows form
def check_db_rows_form(shoulder, elbow, wrist):
    shoulder_elbow_angle = calculate_angle(shoulder, elbow, wrist)

    if shoulder_elbow_angle > 160:  # Arms are not fully extended
        return "Extend your arms fully"
    elif shoulder_elbow_angle < 90:  # Arms are too close to the body
        return "Move your arms away from your body"
    else:
        return "Great form!"

# Main function for dumbbell rows
def db_rows():
    cap = cv2.VideoCapture(0)
    counter = 0 
    stage = None

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

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1],
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0]]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]]

            # Render angle and form feedback
            form_feedback = check_db_rows_form(shoulder, elbow, wrist)
            cv2.putText(image, form_feedback, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (135, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

            # Increment counter based on form feedback
            if form_feedback == "Great form!":
                counter += 1
                print("Rep count:", counter)

        except Exception as e:
            print(e)

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
    return Response(db_rows(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', port=5002, debug=True)
