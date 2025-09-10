import whisper
import sounddevice as sd
import numpy as np
from pynput import keyboard

SAMPLE_RATE = 16000
model = whisper.load_model("base")  # can be "small", "medium", "large"

recording = []
is_recording = False
stream = None

def callback(indata, frames, time, status):
    if status:
        print(status)
    if is_recording:
        recording.append(indata.copy())

def start_recording():
    global is_recording, recording, stream
    if not is_recording:
        print("🎤 Recording started... (press 'q' to stop)")
        recording = []
        is_recording = True
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)
        stream.start()

def stop_recording():
    global is_recording, stream
    if is_recording:
        is_recording = False
        stream.stop()
        stream.close()
        print("🛑 Recording stopped, transcribing...")
        audio = np.concatenate(recording, axis=0).flatten()
        result = model.transcribe(audio)
        print("You said:", result["text"])

def on_press(key):
    try:
        if key.char == 's':  # press s to start
            start_recording()
        elif key.char == 'q':  # press q to stop
            stop_recording()
    except AttributeError:
        if key == keyboard.Key.esc:  # esc to quit
            print("👋 Exiting...")
            return False

print("Press 's' to start, 'q' to stop, 'esc' to quit.")
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
