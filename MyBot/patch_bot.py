--- bot.py	2023-10-25 10:00:00.000000000 +0000
+++ bot.py.new	2023-10-25 10:00:00.000000000 +0000
@@ -67,23 +67,25 @@
 
         loop = asyncio.get_running_loop()
 
-        # Get all sentences from streaming AI
-        sentences = await loop.run_in_executor(
-            None,
-            lambda: list(self.brain.stream_response(user_text))
-        )
+        queue = asyncio.Queue()
+        def producer():
+            try:
+                for sentence in self.brain.stream_response(user_text):
+                    if self.voice.interrupted:
+                        break
+                    loop.call_soon_threadsafe(queue.put_nowait, sentence)
+            finally:
+                loop.call_soon_threadsafe(queue.put_nowait, None)
+
+        producer_task = loop.run_in_executor(None, producer)
 
         interrupted_text = None
         monitor_task     = asyncio.create_task(self.watch_for_interrupt())
 
-        for sentence in sentences:
-            if not sentence.strip() or self.voice.interrupted:
+        while True:
+            sentence = await queue.get()
+            if sentence is None or not sentence.strip() or self.voice.interrupted:
                 break
 
             print(f"Gizmo: {sentence}", flush=True)
