import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import math

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

# Gets the ankle height relative to hip height
def get_kick_height(landmarks, side="right"):
    # Getting the y values for the hip and ankle
    if side == "right":
        ankle_y = landmarks[28].y
        hip_y = landmarks[24].y
    else:
        ankle_y = landmarks[27].y
        hip_y = landmarks[23].y

    # Calculating the relative height and returning it
    # Note: MediaPipe y axis is inverted, so that 0 is at the top of the frame
    return (hip_y - ankle_y)

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
# This will be inbetween the shoulders (at the chest)
def get_chest_height_threshold(landmarks):
    # Getting y values for hte shoulders
    left_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y

    # Average y of shoulders
    chest_y = (left_shoulder_y + right_shoulder_y) / 2

    # Return this chest height as threshold
    return chest_y

