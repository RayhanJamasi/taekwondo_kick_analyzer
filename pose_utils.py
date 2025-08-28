import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import math
import os, json
from datetime import datetime


# Calculates angle at point b given three points. Each point has an 
# x and y value 
def calculate_angle(a, b, c):
    # Converting into NumPy arrays to make the math simpler
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Finding vector of the arms of the angle 
    ba = a - b
    bc = c - b

    # Calculates cosine of the angle using dot formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Getting angle in radians using arcos of cosine value with a range of 
    # [-1.0, 1.0] to help avoid errors from occuring
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    # Converting angle from radians to degrees and returning it
    return np.degrees(angle)

# Gets the position of the hip, knee, and ankle (x, y) depending on 
# if it is the right side or the left side
def get_knee_angle(landmarks, side="right"):
    if side == "right":
        hip = (landmarks[24].x, landmarks[24].y)  
        knee = (landmarks[26].x, landmarks[26].y) 
        ankle = (landmarks[28].x, landmarks[28].y) 
    else:
        hip = (landmarks[23].x, landmarks[23].y)  
        knee = (landmarks[25].x, landmarks[25].y) 
        ankle = (landmarks[27].x, landmarks[27].y) 

    # Returning the angle value between the 3 landmarks
    return calculate_angle(hip, knee, ankle)

# Gets the ankle height
def get_kick_height(landmarks, side="right"):
    # Getting the y values for the hip and ankle
    if side == "right":
        ankle_y = landmarks[28].y
    else:
        ankle_y = landmarks[27].y

    return ankle_y

# Measures the horizontal rotation of the hips by getting the difference in 
# x positions of the right/left sides of the hips. This is important for checking
# the pivoting of kicks
def get_hip_rotation(landmarks):
    # Getting x position of both hips
    left_hip_x = landmarks[23].x
    right_hip_x = landmarks[24].x
    # Returning the difference
    return abs(left_hip_x - right_hip_x)

# Gets the "good height" threshold depending on the height of the person kicking
# This will be inbetween the shoulders (at the chest).
def get_chest_height_threshold(landmarks):    # Get the positions of both sides of shoulders and the mid position
    left_shoulder_y = landmarks[11].y
    right_shoulder_y = landmarks[12].y
    mid_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    # returning the relative chest height
    return mid_shoulder_y

def save_kick_data(frame_buffer, metrics):
    # Creating directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    create_dir = os.path.join("data", f"{timestamp}_{metrics['side']}")
    os.makedirs(create_dir, exist_ok=True)

    # Saving the metrics/info
    metrics_path = os.path.join(create_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Saving the video 
    if frame_buffer:
        height, width, _ = frame_buffer[0].shape
        video_path = os.path.join(create_dir, "kick.mp4")
        # Used for how the video is compressed
        # NOTE: if their are errors in the future, double check this line as this is my first time
                        # trying this
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

        # Finish writing to the file and save/close it 
        for f in frame_buffer:
            output.write(f)
        output.release()

    print(f"Saved the kick data + video in {create_dir}")

