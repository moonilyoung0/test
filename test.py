import cv2
import mediapipe as mp
import numpy as np
import os
import time
from tensorflow.keras.models import load_model


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles # Drawing styles

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(32*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


def test(video_url):
    create_time = time.localtime(time.time())

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join(f'dataset/{create_time.tm_hour}{create_time.tm_min}{create_time.tm_sec}') 

    # Actions that we try to detect
    actions = np.array(['hello', 'happy', 'sad'])

    # Videos are going to be 40 frames in length
    seq_length = 40

    model = load_model('model\model_163201.h5')

    # 1. New detection variables
    threshold = 0.95

    cap = cv2.VideoCapture(video_url)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            sequence = []
            # Loop through video length aka sequence length
            for frame_num in range(seq_length):
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                
            
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)], res)
            break
        cap.release()
        cv2.destroyAllWindows()
        return actions[np.argmax(res)], np.argmax(res)