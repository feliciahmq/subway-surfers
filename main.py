import cv2 as cv
import time
from hand_tracker import HandTracker
from gesture import Gesture
from gesture_to_keyboard import handle_gesture_action

def main():
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

    tracker = HandTracker()
    gesture_detector = Gesture()
    ptime = 0

    if not cap.isOpened():
        print("Camera Frame not detected")
        exit()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = tracker.find_fingers(frame)

        gesture = gesture_detector.recognise_gesture(frame)

        if gesture:
            cv.putText(frame, f"Motion: {gesture}", (10, 50),
                       cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            handle_gesture_action(gesture)
        
        ctime = time.time()
        fps = 1 / (ctime - ptime) if ptime else 0
        ptime = ctime 
        cv.putText(frame, f"FPS: {int(fps)}", (10, 90),
                   cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break   

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
