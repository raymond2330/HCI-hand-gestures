import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
camera = cv2.VideoCapture(0)  # Change the argument to the camera index if you have multiple cameras

# Create a directory for the dataset if it doesn't exist
os.makedirs('dataset', exist_ok=True)

# Define the filename for the CSV file
csv_filename = 'dataset/hand_landmarks.csv'

# Define the columns for the CSV file
columns = [
    'WRIST_X', 'WRIST_Y', 'WRIST_Z',
    'THUMB_CMC_X', 'THUMB_CMC_Y', 'THUMB_CMC_Z',
    'THUMB_MCP_X', 'THUMB_MCP_Y', 'THUMB_MCP_Z',
    'THUMB_IP_X', 'THUMB_IP_Y', 'THUMB_IP_Z',
    'THUMB_TIP_X', 'THUMB_TIP_Y', 'THUMB_TIP_Z',
    'INDEX_FINGER_MCP_X', 'INDEX_FINGER_MCP_Y', 'INDEX_FINGER_MCP_Z',
    'INDEX_FINGER_PIP_X', 'INDEX_FINGER_PIP_Y', 'INDEX_FINGER_PIP_Z',
    'INDEX_FINGER_DIP_X', 'INDEX_FINGER_DIP_Y', 'INDEX_FINGER_DIP_Z',
    'INDEX_FINGER_TIP_X', 'INDEX_FINGER_TIP_Y', 'INDEX_FINGER_TIP_Z',
    'MIDDLE_FINGER_MCP_X', 'MIDDLE_FINGER_MCP_Y', 'MIDDLE_FINGER_MCP_Z',
    'MIDDLE_FINGER_PIP_X', 'MIDDLE_FINGER_PIP_Y', 'MIDDLE_FINGER_PIP_Z',
    'MIDDLE_FINGER_DIP_X', 'MIDDLE_FINGER_DIP_Y', 'MIDDLE_FINGER_DIP_Z',
    'MIDDLE_FINGER_TIP_X', 'MIDDLE_FINGER_TIP_Y', 'MIDDLE_FINGER_TIP_Z',
    'RING_FINGER_MCP_X', 'RING_FINGER_MCP_Y', 'RING_FINGER_MCP_Z',
    'RING_FINGER_PIP_X', 'RING_FINGER_PIP_Y', 'RING_FINGER_PIP_Z',
    'RING_FINGER_DIP_X', 'RING_FINGER_DIP_Y', 'RING_FINGER_DIP_Z',
    'RING_FINGER_TIP_X', 'RING_FINGER_TIP_Y', 'RING_FINGER_TIP_Z',
    'PINKY_MCP_X', 'PINKY_MCP_Y', 'PINKY_MCP_Z',
    'PINKY_PIP_X', 'PINKY_PIP_Y', 'PINKY_PIP_Z',
    'PINKY_DIP_X', 'PINKY_DIP_Y', 'PINKY_DIP_Z',
    'PINKY_TIP_X', 'PINKY_TIP_Y', 'PINKY_TIP_Z',
    'label'
]

# Create the CSV file and write the header
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(columns)

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB and process it using MediaPipe Hands
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    label = None  # Initialize label

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            row_data = []
            for landmark in hand_landmarks.landmark:
                row_data.extend([landmark.x, landmark.y, landmark.z if landmark.z is not None else 0.0])

            # Save data to the CSV file
            with open(csv_filename, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(row_data)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Landmarks', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to quit
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
