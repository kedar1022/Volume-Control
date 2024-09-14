import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, image = webcam.read()
    if not success:
        print("Error: Failed to capture image.")
        break
    
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks for thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates to pixel coordinates
            height, width, _ = image.shape
            thumb_tip = (int(thumb_tip.x * width), int(thumb_tip.y * height))
            index_tip = (int(index_tip.x * width), int(index_tip.y * height))
            
            # Calculate the distance between thumb tip and index tip
            distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            
            # Map the distance to volume control
            # Adjust these values as per your requirements
            min_distance = 30
            max_distance = 200
            min_volume = 0
            max_volume = 100
            
            volume = np.interp(distance, [min_distance, max_distance], [min_volume, max_volume])
            pyautogui.press("volumedown")  # To reset volume to zero initially
            pyautogui.press("volumeup", presses=int(volume/2))  # Set the volume
            
            # Draw the distance on the image
            cv2.putText(image, f'Volume: {int(volume)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow("Hand volume control using python", image)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV window
webcam.release()
cv2.destroyAllWindows()
