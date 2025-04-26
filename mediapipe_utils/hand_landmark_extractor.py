import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.5)

def extract_landmark(image):
    """
    Extract 21 hand landmarks from an image frame using MediaPipe.
    Returns a list of 63 values (x, y, z for each point) or None if no hand detected.
    """
    img_rgb = image[:, :, ::-1]
    result = hands.process(img_rgb)
    if (hand_landmarks:=result.multi_hand_landmarks):
        hand_landmarks = hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    else:
        return None