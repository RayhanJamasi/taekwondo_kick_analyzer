import cv2
import mediapipe as mp
from pose_utils import get_knee_angle, get_kick_height, get_hip_rotation, get_chest_height_threshold
import time

# Setup
mp_drawing = mp.solutions.drawing_utils # Helper functions to draw dots and lines on body
mp_pose = mp.solutions.pose # Detection model that figures out where each part of the body is 

# Creates the pose detection model, show_coords = False to not print out coordinates
# and visibility threshold used for when to print out the debugging info for joints
def run_pose_detection(show_coords=False, visibility_threshold=0.5):
    # Start webcam (0 so it defaults to base webcam)
    cap = cv2.VideoCapture(0)  

    # Kick state tracking variables for right and left sides
    right_kick_in_progress = False
    right_max_knee_angle = 0
    right_max_kick_height = 0
    right_last_kick_end_time = 0
    right_cooldown = 0.5  


    left_kick_in_progress = False
    left_max_knee_angle = 0
    left_max_kick_height = 0
    left_last_kick_end_time = 0
    left_cooldown = 0.5

    # Threshold variables
    kick_height_threshold = 0.1 # For when foot is on the ground
    chest_height_threshold = None # Used to get chest height
    chest_dot_position = None

    start_time = time.time()
    countdown_seconds = 10

    # Load bodytracking system and declaring confidence levels
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # While camera is open
        while cap.isOpened():
            # Get camera status and actual image and break if camera is offline
            ret, frame = cap.read()
            if not ret:
                break

            time_past = time.time() - start_time
            remaining = countdown_seconds - int(time_past)

            if remaining > 0:
                # Countdown text in middle of screen
                h, w, _ = frame.shape
                cv2.putText(frame, str(remaining), (w // 2 - 50, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 8)

                cv2.imshow('Kick Analyzer', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                continue

            # Converting image to RGB for mediapipe
            # Making the flag false for now so its faster
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Mediapipe finds the joints and stores as result
            results = pose.process(image)

            # Converting back to BGR for OpenCV so it can display it correctly
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Checks if any landmarks were found and draws green dots and connecting lines
            # to make a simple stick person of the user
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # if show_coords is True, stores the coordinates of specific body parts as
                # coordinates and printing it out. 
                if show_coords:
                    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

                    # Checking to visibility of both sides of body for kicks
                    right_visible = (
                        right_hip.visibility > visibility_threshold and
                        right_knee.visibility > visibility_threshold and
                        right_ankle.visibility > visibility_threshold
                    )

                    left_visible = (
                        left_hip.visibility > visibility_threshold and
                        left_knee.visibility > visibility_threshold and
                        left_ankle.visibility > visibility_threshold
                    )

                # Getting the current time for the cooldown calculations
                current_time = time.time()

                # Check visibility for left/right ankles and nose
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]

                # Checking if they are both visible
                feet_visible = (left_ankle.visibility > visibility_threshold and 
                                right_ankle.visibility > visibility_threshold)
                head_visible = (nose.visibility > visibility_threshold)

                # Calculate chest height threshold if not set yet and entire body is in frame
                if chest_height_threshold is None and feet_visible and head_visible:                        
                    chest_height_threshold = get_chest_height_threshold(landmarks)
                    print(f"Chest height baseline set: {chest_height_threshold:.3f}")

                    # Getting frame height and width for the screen
                    frame_height, frame_width, _ = image.shape

                    # Get both shoulders position
                    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

                    # Convert x positions to pixels
                    middle_dot_x = int((right_shoulder.x + left_shoulder.x) / 2 * frame_width)

                    # Turn the chest height threshold to pixels using frame height
                    dot_y = int(chest_height_threshold * frame_height)

                    # Getting static position for the dot
                    chest_dot_position = (middle_dot_x, dot_y)

                if chest_dot_position is not None:
                    cv2.circle(image, chest_dot_position, 6, (0, 255, 0), -1)

                # If right side is visible, get the angles using the helper functions
                if right_visible:
                    right_knee_angle = get_knee_angle(landmarks, side="right")
                    right_kick_height = get_kick_height(landmarks, side="right")

                    # If the kick has started and cooldown time ended
                    if (not right_kick_in_progress and
                            right_kick_height > kick_height_threshold and
                            (current_time - right_last_kick_end_time) > right_cooldown):
                        # Kick started
                        right_kick_in_progress = True
                        right_max_knee_angle = right_knee_angle
                        right_max_kick_height = right_kick_height

                    elif right_kick_in_progress:
                        # Updating max values
                        right_max_knee_angle = max(right_max_knee_angle, right_knee_angle)
                        right_max_kick_height = max(right_max_kick_height, right_kick_height)

                        # Kick ended
                        if right_kick_height < kick_height_threshold:
                            right_kick_in_progress = False
                            right_last_kick_end_time = current_time

                            # Qualitative feedback
                            if right_max_kick_height > chest_height_threshold:
                                height_feedback = "Good height!"
                            else:
                                height_feedback = "Try raising your knee higher."
                            knee_feedback = "Nice knee extension!" if right_max_knee_angle > 160 else "Try to straighten your knee more."

                            print(f"""Right Kick Summary:
                                    Max Knee Angle: {right_max_knee_angle:.2f}°
                                    Max Kick Height: {right_max_kick_height:.2f}
                                    Feedback:
                                    - {height_feedback}
                                    - {knee_feedback}
                            """)

                # If left side is visible, get the angles using the helper functions
                if left_visible:
                    left_knee_angle = get_knee_angle(landmarks, side="left")
                    left_kick_height = get_kick_height(landmarks, side="left")

                    if (not left_kick_in_progress and
                            left_kick_height > kick_height_threshold and
                            (current_time - left_last_kick_end_time) > left_cooldown):
                        # Kick started
                        left_kick_in_progress = True
                        left_max_knee_angle = left_knee_angle
                        left_max_kick_height = left_kick_height

                    elif left_kick_in_progress:
                        # Updating the max values
                        left_max_knee_angle = max(left_max_knee_angle, left_knee_angle)
                        left_max_kick_height = max(left_max_kick_height, left_kick_height)

                        # Kick ended
                        if left_kick_height < kick_height_threshold:
                            left_kick_in_progress = False
                            left_last_kick_end_time = current_time

                            # Qualitative feedback
                            if left_max_kick_height > chest_height_threshold:
                                height_feedback = "Good height!"
                            else:
                                height_feedback = "Try raising your knee higher."

                            # If the knee angle is
                            knee_feedback = "Nice knee extension!" if left_max_knee_angle > 160 else "Try to straighten your knee more."

                            print(f"""Left Kick Summary:
                                    Max Knee Angle: {left_max_knee_angle:.2f}°
                                    Max Kick Height: {left_max_kick_height:.2f}
                                    Feedback:
                                    - {height_feedback}
                                    - {knee_feedback}
                            """)

            # Showing the webcam feed
            cv2.imshow('Pose Detection', image)

            # Checks every 10 ms and quits when user presses the q key
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Releasing webcam and closing OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Running the program if called with show_coords to True
if __name__ == "__main__":
    run_pose_detection(show_coords=True)