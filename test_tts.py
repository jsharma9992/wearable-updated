import time
from interaction.tts_manager import TTSManager

def test_tts():
    print("Initializing TTS Manager...")
    tts = TTSManager()
    tts.start()
    
    print("Queueing tests...")
    tts.say("This is a test of the text to speech manager.")
    tts.say("It should work on both Windows and macOS natively.")
    
    print("Waiting for speech to finish...")
    time.sleep(1) # give it time to start
    while tts.is_speaking or tts.queue_size > 0:
        time.sleep(0.5)
        
    print("Stopping...")
    tts.stop()

if __name__ == "__main__":
    test_tts()
