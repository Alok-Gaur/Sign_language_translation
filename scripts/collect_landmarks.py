import os
import sys
import keyboard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
import numpy as np
from mediapipe_utils.hand_landmark_extractor import extract_landmark
from mediapipe_utils.general_utils import create_directory

SAMPLE_PER_CLASS = 50
BASE_DIR = 'data'
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

os.makedirs(BASE_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("Starting data collection...")
    for label in CLASSES:
        print("\nGet ready for sign: '{}'\nPress any key to start capturing".format(label))
        time.sleep(2)

        create_directory(label) #Create directory for current label
        keyboard.read_key()     #Pause until press any key (for preperation of pose)
        
        collected = 0
        while collected < SAMPLE_PER_CLASS:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            landmark = extract_landmark(frame)

            if landmark:
                landmark_path = os.path.join(BASE_DIR, label, f"{collected}.npy")
                np.save(landmark_path, landmark)
                collected += 1
                print(f"Saved sign {collected}/{SAMPLE_PER_CLASS}")
        
            cv2.putText(frame, f"Letter: {label} ({collected}/{SAMPLE_PER_CLASS})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"Collecting Alphabet: {label}", frame)

            if cv2.waitKey(1) & 0xff== ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Terminating the program....")
                return

        
main()