import pyttsx3
import threading

class Voice:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Set speech rate
        self.engine.setProperty('volume', 1.0)  # Set volume level (0.0 to 1.0)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()