import os
import sys
import asyncio
import time
import collections
import pygame

from face  import BotFace
from brain import BotBrain
from voice import BotVoice
from ears  import BotEars
from mind  import BotMind


class CompanionBot:
    def __init__(self):
        print("Initializing Gizmo...")
        self.brain       = BotBrain()
        self.face        = BotFace()
        self.voice       = BotVoice(voice_name=self.brain.voice)
        self.ears        = BotEars(api_key=self.brain.api_key)
        self.mind        = BotMind(self.brain, self.voice, self.face)
        self.running     = True
        self.is_speaking = False

    async def watch_for_interrupt(self):
        """
        Runs while Gizmo speaks.
        _volume is ALWAYS current from the reader thread — works even during playback.
        Uses rolling window to avoid echo false triggers.
        """
        WINDOW   = 6
        NEEDED   = 4
        window   = collections.deque(maxlen=WINDOW)

        # Sample echo level for first 0.5s to set adaptive threshold
        echo_samples = []
        deadline = time.time() + 0.5
        while time.time() < deadline:
            echo_samples.append(self.ears.get_current_volume())
            await asyncio.sleep(0.05)
        echo_peak  = max(echo_samples) if echo_samples else self.ears.THRESHOLD
        thresh     = max(echo_peak * 2.5, self.ears.THRESHOLD * 1.5)

        while self.is_speaking:
            vol = self.ears.get_current_volume()
            window.append(vol)

            if len(window) == WINDOW:
                loud = sum(1 for v in window if v > thresh)
                if loud >= NEEDED:
                    print("\n[Interrupted!]", flush=True)
                    self.voice.stop()
                    self.ears.stop_recording()
                    self.face.state = "surprised"
                    await asyncio.sleep(0.2)
                    text = await self.ears.capture_interrupt()
                    return text

            await asyncio.sleep(0.03)
        return None

    async def respond(self, user_text):
        self.is_speaking       = True
        self.voice.interrupted = False
        self.face.state        = "thinking"

        loop = asyncio.get_running_loop()

        # Stream sentences from AI as they are generated
        queue = asyncio.Queue()

        def producer():
            try:
                for sentence in self.brain.stream_response(user_text):
                    if self.voice.interrupted:
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, sentence)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer_task = loop.run_in_executor(None, producer)

        audio_queue = asyncio.Queue()
        async def tts_worker():
            try:
                while True:
                    sentence = await queue.get()
                    if sentence is None or not sentence.strip() or self.voice.interrupted:
                        await audio_queue.put((None, None))
                        break
                    emotion = self.brain.detect_emotion(sentence)
                    temp_path = await self.voice.generate_temp_audio(sentence, emotion=emotion)
                    await audio_queue.put((sentence, temp_path))
                    if self.voice.interrupted:
                        break
            except asyncio.CancelledError:
                pass

        tts_task = asyncio.create_task(tts_worker())

        interrupted_text = None
        monitor_task     = asyncio.create_task(self.watch_for_interrupt())

        while True:
            sentence, temp_path = await audio_queue.get()
            if sentence is None or self.voice.interrupted:
                break

            print(f"Gizmo: {sentence}", flush=True)
            completed = await self.voice.play_temp_audio(temp_path, face=self.face)

            if not completed or self.voice.interrupted:
                break
        tts_task.cancel()

        # Cleanup unplayed temp files
        import os
        while not audio_queue.empty():
            _, path = audio_queue.get_nowait()
            if path and os.path.exists(path):
                try: os.remove(path)
                except: pass

        # Cancel monitor and get interrupt result
        if not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            interrupted_text = None
        else:
            interrupted_text = monitor_task.result()

        self.is_speaking       = False
        self.voice.interrupted = False
        self.mind.last_interaction = time.time()

        if len(self.brain.history) % 10 == 0:
            loop.run_in_executor(None, self.brain.extract_and_save_memory)

        if interrupted_text and len(interrupted_text.strip().replace(".", "").replace(" ", "")) >= 2:
            print(f"\nYou (cut in): {interrupted_text}", flush=True)
            await self.respond(interrupted_text)

    async def conversation_loop(self):
        while self.running:
            if self.face.state == "sleeping":
                await asyncio.sleep(0.5)
                continue
            if self.is_speaking:
                await asyncio.sleep(0.1)
                continue

            user_text = await self.ears.listen_and_record()

            if user_text and len(user_text.strip().replace(".", "").replace(" ", "")) >= 2:
                print(f"\nYou: {user_text}", flush=True)
                self.face.state = "surprised"
                await asyncio.sleep(0.1)
                await self.respond(user_text)

            await asyncio.sleep(0.02)

    async def do_greeting(self):
        await asyncio.sleep(1.5)
        try:
            loop     = asyncio.get_running_loop()
            greeting = await loop.run_in_executor(
                None,
                lambda: self.brain.get_response(
                    "[System: You just woke up. Greet your owner warmly in one short sentence.]"
                )
            )
            print(f"Gizmo: {greeting}", flush=True)
            emotion = self.brain.detect_emotion(greeting)
            await self.voice.speak(greeting, face=self.face, emotion=emotion)
        except Exception as e:
            print(f"[Greeting error: {e}]")

    async def main_loop(self):
        print("\n=== Gizmo is Awake ===")
        print("Talk anytime — cut her off mid-sentence, she'll stop and listen.\n")

        asyncio.create_task(self.do_greeting())
        self.mind.start_scheduler(asyncio.get_running_loop())
        asyncio.create_task(self.conversation_loop())

        while self.face.running:
            self.face.update()
            self.mind.run_pending()
            await asyncio.sleep(0.01)

        self.running = False
        self.brain.save_memory()
        print("\n[Goodbye]")


if __name__ == "__main__":
    bot = CompanionBot()
    try:
        asyncio.run(bot.main_loop())
    except KeyboardInterrupt:
        print("\n[Saving and closing...]")
        bot.brain.save_memory()
    finally:
        pygame.quit()