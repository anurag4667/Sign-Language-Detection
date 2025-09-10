import cv2
import mediapipe as mp
import numpy as np
import pickle
import whisper
import sounddevice as sd
from pynput import keyboard

# ------------------------
# Load models
# ------------------------
# Load Whisper
SAMPLE_RATE = 16000
whisper_model = whisper.load_model("base")

# Load RF Model for hand gestures
with open("./rf_model.p", "rb") as f:
    model = pickle.load(f)
rf_model = model["model"]

labels = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9"}

# ------------------------
# Mediapipe Hand setup
# ------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9)

# ------------------------
# Audio Recording setup
# ------------------------
recording = []
is_recording = False
stream = None

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    if is_recording:
        recording.append(indata.copy())

def start_recording():
    global is_recording, recording, stream
    if not is_recording:
        print("🎤 Recording started... (press 'a' to stop)")
        recording = []
        is_recording = True
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback)
        stream.start()

def stop_recording():
    global is_recording, stream
    if is_recording:
        is_recording = False
        stream.stop()
        stream.close()
        print("🛑 Recording stopped, transcribing...")
        audio = np.concatenate(recording, axis=0).flatten()
        result = whisper_model.transcribe(audio)
        print("You said:", result["text"])

# ------------------------
# Key handling
# ------------------------
def on_press(key):
    try:
        if key.char == 's':  # start speech recording
            start_recording()
        elif key.char == 'a':  # stop and transcribe
            stop_recording()
    except AttributeError:
        if key == keyboard.Key.esc or key == keyboard.KeyCode.from_char('q'):
            print("👋 Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ------------------------
# Video Loop for Hand Signs
# ------------------------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_image = hands.process(frame_rgb)
    hand_landmarks = processed_image.multi_hand_landmarks

    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_coordinates, y_coordinates, normalized_landmarks = [], [], []
            for lm in hand_landmark.landmark:
                x_coordinates.append(lm.x)
                y_coordinates.append(lm.y)
            min_x, min_y = min(x_coordinates), min(y_coordinates)

            for lm in hand_landmark.landmark:
                normalized_landmarks.extend((lm.x - min_x, lm.y - min_y))

            x1, y1 = int(min(x_coordinates) * width), int(min(y_coordinates) * height)
            x2, y2 = int(max(x_coordinates) * width), int(max(y_coordinates) * height)

            sample = np.asarray(normalized_landmarks).reshape(1, -1)
            pred = rf_model.predict(sample)
            predicted_character = labels[int(pred[0])]

            cv2.rectangle(frame, (x1 + 10, y1 + 10), (x2, y2), (100, 200, 100), 4)
            cv2.putText(frame, predicted_character, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Video Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
