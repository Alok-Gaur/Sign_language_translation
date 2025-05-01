import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
import string
import tensorflow as tf
import mediapipe as mp
import numpy as np
from collections import deque


class SignLanguageTranslator:
    def __init__(self, model_path = "models/sign_language_model.keras", buffer_size=15):
        self.model = tf.keras.models.load_model(model_path)
        self.label = list(string.ascii_uppercase)
        self.buffer_size = buffer_size
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.current_sentence = ''

        #MediaPipe Drawing Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_landmarks(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    
    def predict_letter(self, landmarks):
        landmarks = np.array([landmarks])
        prediction = self.model.predict(landmarks, verbose=1)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        return self.label[predicted_class], confidence
    
    def update_sentence(self, predicted_letter):
        self.prediction_buffer.append(predicted_letter)

        if self.prediction_buffer.count(predicted_letter) > self.buffer_size//2:
            if not self.current_sentence.endswith(predicted_letter):
                self.current_sentence += predicted_letter
    
    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, 
                                                   hand_landmarks, 
                                                   self.mp_hands.HAND_CONNECTIONS)
                    
                    landmarks = self.extract_landmarks(hand_landmarks)
                    predicted_letter, confidence = self.predict_letter(landmarks)

                    if confidence>0.8:
                        self.update_sentence(predicted_letter)
                    
                    cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Sentence:{self.current_sentence}",
                        (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow("Sign Language Translator", frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            elif cv2.waitKey(1) & 0xff == ord('c'):
                self.current_sentence = ''
                self.prediction_buffer.clear()
        cap.release()
        cv2.destroyAllWindows()
        

new_class=SignLanguageTranslator().run()