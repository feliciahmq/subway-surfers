import cv2 as cv
import mediapipe as mp
import math

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_fingers(self, frame, draw=True):
        imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # opencv reads in BGR, mediapipe reads in RGB
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks and draw:
            for hand_landmark in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame,
                                            hand_landmark,
                                            self.mp_hands.HAND_CONNECTIONS)
        return frame
    
    def get_landmarks(self, frame, hand_number=0, draw=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            h, w, c = frame.shape

            for i, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append((i, cx, cy))
                if draw:
                    cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return self.landmark_list

    def get_distance(self, p1, p2, landmarks, frame, draw=True):
        x1, y1 = landmarks[p1][1], landmarks[p1][2]
        x2, y2 = landmarks[p2][1], landmarks[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(frame, (x1, y1),  (x2, y2), (255, 0, 255), 3)
            cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(frame, (x2, y2), 15, (255, 0, 0), cv.FILLED)
            cv.circle(frame, (cx, cy), 15, (0, 0, 255), cv.FILLED)

        distance = math.hypot(x2 - x1, y2 - y1)
        return distance