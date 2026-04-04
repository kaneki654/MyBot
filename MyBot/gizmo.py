import asyncio
import shlex

def get_ai_response(user_input, feedback=None):
    """Mock function representing the AI."""
    if not feedback:
        return "SPEAK: I'll make that folder for you.\nDO: mkdir -p ~/Music/Gizmo_playlist"
    else:
        # Provide a comment based on the feedback
        safe_feedback = feedback.replace('\n', ' ')[:50]
        if not safe_feedback:
            safe_feedback = "Command succeeded with no output."
        return f"SPEAK: The command finished. The output was: {safe_feedback}\nDO: none"

def parse_response(response_text):
    """Parses the AI's response to extract speak and do strings."""
    speak = ""
    do_cmd = "none"
    for line in response_text.splitlines():
        if line.startswith("SPEAK: "):
            speak = line[7:].strip()
        elif line.startswith("DO: "):
            do_cmd = line[4:].strip()
    return speak, do_cmd

async def run_command(command):
    """Runs a bash command asynchronously and captures its stdout/stderr."""
    if command == "none" or not command:
        return ""
    
    print(f"[Terminal] Executing: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    output = ""
    if stdout:
        output += stdout.decode()
    if stderr:
        output += stderr.decode()
        
    return output.strip()

async def speak_text(text):
    """Runs TTS asynchronously using edge-tts and mpv."""
    if not text:
        return
        
    print(f"[Gizmo] Speaking: {text}")
    escaped_text = shlex.quote(text)
    
    # Using edge-tts and mpv as the default audio player
    command = f"edge-tts --text {escaped_text} --write-media speech.mp3 && mpv speech.mp3"
    
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()

async def execute_workflow():
    """Main loop simulating the workflow with sample iterations."""
    sample_inputs = [
        "Can you create a playlist folder for me?",
        "Did you finish creating it?"
    ]
    
    feedback = ""
    
    for i, user_input in enumerate(sample_inputs):
        print(f"\n--- Iteration {i + 1} ---")
        print(f"[User] {user_input}")
        
        # 1. Get AI response
        response = get_ai_response(user_input, feedback)
        print(f"[AI Raw Response]\n{response}")
        
        # 2. Parse response
        speak_str, do_str = parse_response(response)
        
        # 3. Concurrently run command and speak
        print("[System] Starting concurrent execution of Speak and Do...")
        command_task = asyncio.create_task(run_command(do_str))
        speak_task = asyncio.create_task(speak_text(speak_str))
        
        results = await asyncio.gather(command_task, speak_task)
        
        # 4. Capture the output of the command for the next iteration
        feedback = results[0]
        if do_str != "none":
            if feedback:
                print(f"[Terminal Output]\n{feedback}")
            else:
                print("[Terminal Output] (Command finished with no output)")

async def main():
    await execute_workflow()

if __name__ == '__main__':
    asyncio.run(main())