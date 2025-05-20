import pyautogui
import time

last_trigger_time = {}
cooldown = 1

def handle_gesture_action(gesture):
    global last_trigger_time
    now = time.time()

    if gesture is None:
        return

    if gesture in last_trigger_time and (now - last_trigger_time[gesture] < cooldown):
        return

    last_trigger_time[gesture] = now

    if gesture == "open_palm":
        pyautogui.press("up")
    elif gesture == "thumb_down":
        pyautogui.press("down")
    elif gesture == "left":
        pyautogui.press("left")
    elif gesture == "right":
        pyautogui.press("right")
    else:
        print(f"Unknown gesture: {gesture}")
