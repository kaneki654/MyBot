import schedule
import time
import random
import datetime
import asyncio


class BotMind:
    def __init__(self, brain, voice, face):
        self.brain            = brain
        self.voice            = voice
        self.face             = face
        self.last_interaction = time.time()
        self._loop            = None   # set by bot.py after loop starts
        self._scheduled_job   = None

        # Load sleep/wake times from file
        self.sleep_hour, self.sleep_minute, self.wake_hour, self.wake_minute = self._load_schedule()

        self.thought_templates = [
            "You just had an interesting thought about something. Share it in one or two sentences.",
            "You want to ask your owner one curious question about their life or day.",
            "You just thought of something fascinating about the world. Share it.",
            "You want to gently check in on how your owner is doing today.",
            "You noticed something about the time of day. Comment on it in your own way.",
            "You just remembered something that made you smile. Share it.",
            "You had a small feeling — happy, curious, or wistful. Express it briefly.",
        ]

    def _load_schedule(self):
        """Read sleep and wake times from sleep_schedule.txt."""
        import os
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        sched_path = os.path.join(base_dir, "sleep_schedule.txt")

        # Defaults if file missing
        sleep_h, sleep_m = 22, 0
        wake_h,  wake_m  = 7,  0

        try:
            with open(sched_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if len(lines) >= 2:
                sh, sm = lines[0].split(":")
                wh, wm = lines[1].split(":")
                sleep_h, sleep_m = int(sh), int(sm)
                wake_h,  wake_m  = int(wh), int(wm)
                print(f"[Schedule loaded] Sleep: {sleep_h:02d}:{sleep_m:02d} | Wake: {wake_h:02d}:{wake_m:02d}")
            else:
                print("[Schedule] File incomplete, using defaults 22:00 / 07:00")
        except FileNotFoundError:
            print("[Schedule] sleep_schedule.txt not found, using defaults 22:00 / 07:00")
        except Exception as e:
            print(f"[Schedule] Error reading file: {e} — using defaults")

        return sleep_h, sleep_m, wake_h, wake_m

    def _is_sleep_time(self):
        now   = datetime.datetime.now()
        sleep = now.replace(hour=self.sleep_hour, minute=self.sleep_minute, second=0)
        wake  = now.replace(hour=self.wake_hour,  minute=self.wake_minute,  second=0)

        # Handle overnight (e.g. sleep 22:00, wake 07:00)
        if sleep > wake:
            return now >= sleep or now <= wake
        else:
            return sleep <= now <= wake

    def _is_drowsy_time(self):
        """2 hours before sleep time."""
        now       = datetime.datetime.now()
        drowsy_h  = (self.sleep_hour - 2) % 24
        drowsy    = now.replace(hour=drowsy_h, minute=self.sleep_minute, second=0)
        sleep_t   = now.replace(hour=self.sleep_hour, minute=self.sleep_minute, second=0)
        return drowsy <= now < sleep_t

    def check_time_and_state(self):
        if self._is_sleep_time():
            if self.face.state != "sleeping":
                self._fire_coroutine(self.go_to_sleep())
        elif self._is_drowsy_time():
            if self.face.state not in ("sleeping", "drowsy"):
                self.face.state = "drowsy"
                print("[State] Gizmo is drowsy")
        else:
            if self.face.state in ("sleeping", "drowsy"):
                self._fire_coroutine(self.wake_up())

    async def go_to_sleep(self):
        print("[Gizmo going to sleep...]")
        import asyncio
        loop = asyncio.get_running_loop()
        goodnight = await loop.run_in_executor(None, self.brain.get_response, 
            "[System: It is time to sleep. Say a short warm goodnight to your owner in one sentence.]"
        )
        print(f"Gizmo: {goodnight}")
        await self.voice.speak(goodnight, face=self.face, emotion="sleepy")
        self.face.state = "sleeping"

    async def wake_up(self):
        print("[Gizmo waking up...]")
        self.face.state = "idle"
        import asyncio
        await asyncio.sleep(1.5)  # groggy pause
        loop = asyncio.get_running_loop()
        good_morning = await loop.run_in_executor(None, self.brain.get_response,
            "[System: You just woke up from sleeping. Say a short sleepy good morning in one sentence.]"
        )
        print(f"Gizmo: {good_morning}")
        await self.voice.speak(good_morning, face=self.face, emotion="sleepy")

    async def spontaneous_thought(self):
        if self.face.state in ("sleeping",):
            self._reschedule()
            return

        if time.time() - self.last_interaction > 1800:  # 30 mins idle
            prompt  = "You haven't heard from your owner in a while. Share a bored or curious thought in one sentence."
            emotion = "bored"
            self.face.state = "bored"
        else:
            prompt  = random.choice(self.thought_templates)
            emotion = "normal"

        import asyncio
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self.brain.get_response, f"[Internal Thought Prompt: {prompt}]")
        print(f"\nGizmo (spontaneous): {response}")
        await self.voice.speak(response, face=self.face, emotion=emotion)

        # Extract memory every ~5 spontaneous thoughts
        if random.random() < 0.2:
            self.brain.extract_and_save_memory()

        self._reschedule()  # re-randomize the next thought time

    def _fire_coroutine(self, coro):
        """Safely schedule a coroutine onto the running event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            print("[Mind] Event loop not ready yet, skipping coroutine")

    def _reschedule(self):
        """Cancel current job and schedule a new one with a fresh random interval."""
        if self._scheduled_job:
            try:
                schedule.cancel_job(self._scheduled_job)
            except Exception:
                pass

        interval = random.randint(8, 25)
        print(f"[Mind] Next spontaneous thought in {interval} minutes")
        self._scheduled_job = schedule.every(interval).minutes.do(
            lambda: self._fire_coroutine(self.spontaneous_thought())
        )

    def start_scheduler(self, loop):
        """Call this from bot.py after the event loop is running."""
        self._loop = loop
        self._reschedule()
        schedule.every(1).minutes.do(self.check_time_and_state)
        print("[Mind] Scheduler started")

    def run_pending(self):
        schedule.run_pending()