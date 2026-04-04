import os
import wave
import time
import asyncio
import requests
import math
import struct
import threading
import tempfile

try:
    import pyaudio
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class BotEars:
    def __init__(self, api_key):
        self.api_key     = api_key
        self.chunk_size  = 1024
        self.channels    = 1
        self.rate        = 16000
        self.ready       = False
        self.p           = None
        self.stream      = None

        self.THRESHOLD   = 2000
        self.SILENCE_SEC = 0.7
        self.MAX_SEC     = 15

        # Single reader thread state
        self._volume      = 0          # always current
        self._recording   = False      # set True to collect frames
        self._frames      = []         # collected when _recording=True
        self._rec_lock    = threading.Lock()
        self._stop_rec    = threading.Event()  # signal to stop recording

        self._init_mic()

    def _init_mic(self):
        if not HAS_AUDIO:
            return
        try:
            import pyaudio
            self.p = pyaudio.PyAudio()
            print("\n=== Audio Devices ===")
            inputs = []
            for i in range(self.p.get_device_count()):
                d = self.p.get_device_info_by_index(i)
                if d['maxInputChannels'] > 0:
                    inputs.append(i)
                    print(f"ID {i}: {d['name']}")

            def dev_name(i):
                return self.p.get_device_info_by_index(i).get('name', '').lower()

            preferred = sorted(inputs, key=lambda i: (
                0 if 'pulse'    in dev_name(i) else
                1 if 'default'  in dev_name(i) else
                2 if 'pipewire' in dev_name(i) else 3
            ))

            for dev_id in preferred:
                try:
                    s = self.p.open(
                        format=pyaudio.paInt16, channels=self.channels,
                        rate=self.rate, input=True,
                        input_device_index=dev_id,
                        frames_per_buffer=self.chunk_size
                    )
                    s.close()
                    self.stream = self.p.open(
                        format=pyaudio.paInt16, channels=self.channels,
                        rate=self.rate, input=True,
                        input_device_index=dev_id,
                        frames_per_buffer=self.chunk_size
                    )
                    self.ready = True
                    name = self.p.get_device_info_by_index(dev_id)['name']
                    print(f"✓ Microphone: {name} (device {dev_id})")
                    # Start the ONE reader thread
                    threading.Thread(target=self._reader_thread, daemon=True).start()
                    break
                except Exception as e:
                    print(f"  ✗ device {dev_id}: {e}")
        except Exception as e:
            print(f"!! Mic init: {e}")

    def _rms(self, block):
        if not block or len(block) < 2:
            return 0
        try:
            n = len(block) // 2
            s = struct.unpack(f"{n}h", block)
            return math.sqrt(sum(x * x for x in s) / n)
        except Exception:
            return 0

    def _reader_thread(self):
        """
        The ONLY thread that ever reads the mic.
        Always updates self._volume.
        When self._recording is True, also appends to self._frames.
        """
        while True:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                rms  = self._rms(data)
                self._volume = rms

                if self._recording:
                    with self._rec_lock:
                        self._frames.append((rms, data))

            except Exception as e:
                print(f"Exception in reader_thread: {e}")
                time.sleep(0.05)

    def get_current_volume(self):
        """Always returns latest volume — even while bot is speaking."""
        return self._volume

    def _do_record(self):
        """
        Called from executor thread.
        Waits for speech, records until silence, returns frames.
        """
        # Wait for speech
        print(f"[Waiting... threshold={self.THRESHOLD}]", flush=True)
        while True:
            v = self._volume
            if v > self.THRESHOLD:
                print(f"[TRIGGERED at vol={int(v)}]", flush=True)
                break
            time.sleep(0.02)

        # Start collecting
        with self._rec_lock:
            self._frames = []
        self._recording  = True
        self._stop_rec.clear()
        print("[Recording started...]", flush=True)

        silence_start = None
        max_chunks    = int(self.rate / self.chunk_size * self.MAX_SEC)
        last_log      = time.time()

        while True:
            time.sleep(0.02)
            with self._rec_lock:
                frames_copy = list(self._frames)

            # Log every second so we know it's alive
            now = time.time()
            if now - last_log > 1.0:
                print(f"[frames: {len(frames_copy)}, vol: {int(self._volume)}]", flush=True)
                last_log = now

            if len(frames_copy) >= max_chunks:
                print(f"[Max chunks reached: {len(frames_copy)}]", flush=True)
                break

            if frames_copy:
                rms = frames_copy[-1][0]
                if rms < self.THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.SILENCE_SEC:
                        print(f"[Silence detected, frames: {len(frames_copy)}]", flush=True)
                        break
                else:
                    silence_start = None
            else:
                # No frames yet - reader thread might not be running
                if now - last_log < 0.1:
                    print("[WARNING: no frames collecting - reader thread issue?]", flush=True)

            if self._stop_rec.is_set():
                break

        self._recording = False
        with self._rec_lock:
            result = list(self._frames)
            self._frames = []

        print(f"[Recording done, {len(result)} frames]", flush=True)
        return [d for _, d in result]

    def _do_capture_interrupt(self):
        """Record what user says after interrupting."""
        time.sleep(0.1)
        with self._rec_lock:
            self._frames = []
        self._recording  = True
        self._stop_rec.clear()

        silence_start = None
        max_chunks    = int(self.rate / self.chunk_size * self.MAX_SEC)

        while True:
            time.sleep(0.02)
            with self._rec_lock:
                frames_copy = list(self._frames)

            if len(frames_copy) >= max_chunks:
                break

            if frames_copy:
                rms = frames_copy[-1][0]
                if rms < self.THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.SILENCE_SEC:
                        break
                else:
                    silence_start = None

        self._recording = False
        with self._rec_lock:
            result = list(self._frames)
            self._frames = []

        return [d for _, d in result]

    def _transcribe(self, frames):
        if len(frames) < 5:
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                path = tmp.name
            import pyaudio
            wf = wave.open(path, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            url     = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            for _ in range(3):
                try:
                    with open(path, 'rb') as f:
                        r = requests.post(url, headers=headers, files={
                            "file":     (os.path.basename(path), f, "audio/wav"),
                            "model":    (None, "whisper-large-v3-turbo"),
                            "language": (None, "en"),
                        }, timeout=15)
                        r.raise_for_status()
                        text = r.json().get("text", "").strip()
                        return text if text else None
                except Exception:
                    time.sleep(0.5)
            return None
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def stop_recording(self):
        """Signal recording to stop early (called on interrupt)."""
        self._stop_rec.set()
        self._recording = False

    async def listen_and_record(self):
        if not self.ready:
            await asyncio.sleep(0.5)
            return None
        loop = asyncio.get_running_loop()
        frames = await loop.run_in_executor(None, self._do_record)
        return await loop.run_in_executor(None, self._transcribe, frames)

    async def capture_interrupt(self):
        loop = asyncio.get_running_loop()
        frames = await loop.run_in_executor(None, self._do_capture_interrupt)
        return await loop.run_in_executor(None, self._transcribe, frames)

    def __del__(self):
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except Exception:
            pass
