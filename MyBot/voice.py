import edge_tts
import asyncio
import pygame
import os
import threading


class BotVoice:
    def __init__(self, voice_name="en-US-AriaNeural"):
        self.voice       = voice_name
        self.base_dir    = os.path.dirname(os.path.abspath(__file__))
        self.temp_file   = os.path.join(self.base_dir, "temp_audio.mp3")
        self.interrupted = False
        self._lock       = threading.Lock()

    async def generate_audio(self, text, rate="+0%", pitch="+0Hz"):
        communicate = edge_tts.Communicate(text, self.voice, rate=rate, pitch=pitch)
        await communicate.save(self.temp_file)

    def stop(self):
        """Stop playback immediately from any thread."""
        self.interrupted = True
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def play_audio_sync(self):
        """
        Plays audio in a blocking way but checks interrupted flag every 30ms.
        Returns True = finished naturally, False = was cut short.
        """
        if not os.path.exists(self.temp_file):
            return True
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self.interrupted:
                    pygame.mixer.music.stop()
                    return False
                pygame.time.wait(30)
            return True
        except Exception as e:
            err = str(e).lower()
            if "corrupt" not in err and "bad" not in err:
                print(f"Audio error: {e}")
            return True
        finally:
            try:
                if os.path.exists(self.temp_file):
                    os.remove(self.temp_file)
            except Exception:
                pass

    async def speak(self, text, face=None, emotion="normal"):
        rate, pitch = "+0%", "+0Hz"
        if   emotion == "sleepy":    rate, pitch = "-25%", "-4Hz"
        elif emotion == "excited":   rate, pitch = "+18%", "+5Hz"
        elif emotion == "happy":     rate, pitch = "+10%", "+3Hz"
        elif emotion == "sad":       rate, pitch = "-15%", "-3Hz"
        elif emotion == "bored":     rate, pitch = "-12%", "-2Hz"
        elif emotion == "surprised": rate, pitch = "+20%", "+6Hz"
        elif emotion == "sincere":   rate, pitch = "-8%",  "-1Hz"

        if text.strip().endswith('?'):
            current_pitch = int(pitch.replace("Hz", ""))
            current_pitch += 2
            pitch = f"{current_pitch:+d}Hz"

        if self.interrupted:
            return False

        if face:
            face.state = "thinking"

        try:
            await self.generate_audio(text, rate, pitch)
        except Exception as e:
            print(f"TTS error: {e}")
            if face: face.state = "idle"
            return False

        if self.interrupted:
            if face: face.state = "idle"
            return False

        if face:
            face.state = "talking"

        # Run blocking playback in executor so event loop stays free
        loop = asyncio.get_event_loop()
        completed = await loop.run_in_executor(None, self.play_audio_sync)

        if face:
            face.state = "idle"

        return completed
    async def generate_temp_audio(self, text, emotion="normal"):
        import tempfile
        rate, pitch = "+0%", "+0Hz"
        if   emotion == "sleepy":    rate, pitch = "-25%", "-4Hz"
        elif emotion == "excited":   rate, pitch = "+18%", "+5Hz"
        elif emotion == "happy":     rate, pitch = "+10%", "+3Hz"
        elif emotion == "sad":       rate, pitch = "-15%", "-3Hz"
        elif emotion == "bored":     rate, pitch = "-12%", "-2Hz"
        elif emotion == "surprised": rate, pitch = "+20%", "+6Hz"
        elif emotion == "sincere":   rate, pitch = "-8%",  "-1Hz"

        if text.strip().endswith('?'):
            current_pitch = int(pitch.replace("Hz", ""))
            current_pitch += 2
            pitch = f"{current_pitch:+d}Hz"

        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=rate, pitch=pitch)
            await communicate.save(path)
            return path
        except Exception as e:
            print(f"TTS error: {e}")
            try:
                os.remove(path)
            except: pass
            return None

    def play_audio_sync_path(self, path):
        if not path or not os.path.exists(path):
            return True
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self.interrupted:
                    pygame.mixer.music.stop()
                    return False
                pygame.time.wait(30)
            return True
        except Exception as e:
            err = str(e).lower()
            if "corrupt" not in err and "bad" not in err:
                print(f"Audio error: {e}")
            return True
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    async def play_temp_audio(self, path, face=None):
        if not path:
            return False
        if self.interrupted:
            return False
        if face:
            face.state = "talking"

        loop = asyncio.get_event_loop()
        completed = await loop.run_in_executor(None, self.play_audio_sync_path, path)

        if face:
            face.state = "idle"

        return completed
