import mediapipe as mp
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class Gesture:
    def __init__(self):
        model_path = 'gesture_recognizer.task'
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.recognizer = GestureRecognizer.create_from_options(options)
        self.last_gesture = None
        self.last_time = 0
        self.cooldown = 1.0

    def recognise_gesture(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.recognizer.recognize(mp_image)

        if result.gestures:
            gesture_name = result.gestures[0][0].category_name
            if gesture_name != "None":
                return gesture_name.lower()

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            finger_gesture = self.detect_left_right_from_landmarks(landmarks)
            if finger_gesture:
                return finger_gesture
    
        return None

    def detect_left_right_from_landmarks(self, landmarks):
        if len(landmarks) < 21:
            return None

        wrist_x = landmarks[0].x

        index_tip_x = landmarks[8].x
        index_tip_y = landmarks[8].y
        index_pip_y = landmarks[6].y

        middle_tip_y = landmarks[12].y
        middle_pip_y = landmarks[10].y

        ring_tip_y = landmarks[16].y
        ring_pip_y = landmarks[14].y

        pinky_tip_y = landmarks[20].y
        pinky_pip_y = landmarks[18].y

        # only index, middle fingers area pointing up
        index_up = index_tip_y < index_pip_y - 0.01
        middle_up = middle_tip_y < middle_pip_y - 0.01
        ring_down = ring_tip_y > ring_pip_y + 0.01
        pinky_down = pinky_tip_y > pinky_pip_y + 0.01

        if index_up and middle_up and ring_down and pinky_down:
            if index_tip_x < wrist_x - 0.05:
                return "right"
            elif index_tip_x > wrist_x + 0.05:
                return "left"

        return None

