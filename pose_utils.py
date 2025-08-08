import cv2
import mediapipe as mp

# Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Creates the pose detection model, show_coords = False to not print out coordinates
# and visibility threshold used for when to print out the debugging info for joints
def run_pose_detection(show_coords=False, visibility_threshold=0.5):
    # Start webcam (0 so it defaults to base webcam)
    cap = cv2.VideoCapture(0)  

    # Load bodytracking system and declaring confidence levels
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # While camera is open
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

            # Checks if any landmarks were found and draws green dots and connecting lines
            # to make a simple stick person of the user
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # if show_coords is True, stores the coordinates of specific body parts as
                # coordinates and printing it out. 
                if show_coords:
                    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]


                    if (right_hip.visibility > visibility_threshold and
                        right_knee.visibility > visibility_threshold and
                        right_ankle.visibility > visibility_threshold):
                        print(f"Hip: ({right_hip.x:.2f}, {right_hip.y:.2f}) | "
                              f"Knee: ({right_knee.x:.2f}, {right_knee.y:.2f}) | "
                              f"Ankle: ({right_ankle.x:.2f}, {right_ankle.y:.2f})")


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
