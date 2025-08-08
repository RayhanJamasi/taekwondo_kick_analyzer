import cv2
import mediapipe as mp

# Setup
mp_drawing = mp.solutions.drawing_utils # Helper functions to draw dots and lines on body
mp_pose = mp.solutions.pose # Detection model that figures out where each part of the body is 

# Start webcam (0 so it defaults to base webcam)
cap = cv2.VideoCapture(0)

# Load bodytracking system and declaring confidence levels
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    # While webcam is online
    while cap.isOpened():
        # Get camera status and actual image and break if camera is offline
        ret, frame = cap.read()
        if not ret:
            break

        # Converting image to RGB for mediapipe
        # Making the flag false for now so its faster
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Mediapipe finds the joints and stores as result
        results = pose.process(image)

        # Converting back to BGR for OpenCV so it can display it correctly
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Connects the dots in results with an overlay of lines to make a skeleton
        # Green dots mean bigger joints while red dots mean smaller joints
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        # Showing the webcam feed
        cv2.imshow('Taekwondo Kick Analyzer - Pose Tracking', image)

        # Checks every 10 ms and quits when user presses the q key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Releasing webcam and closing openCV windows
cap.release()
cv2.destroyAllWindows()
