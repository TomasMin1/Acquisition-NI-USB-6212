import time
import subprocess
from datetime import datetime, time as dtime

# Horarios para playback
tstart_playback = dtime(14, 22, 0)
tend_playback   = dtime(14, 25, 0)

# Comandos
playback_cmd = ['python', 'Global code/playback_global.py']
acquisition_cmd = ['python', 'Global code/adquisicion_global.py']

# Estado actual
current_process = None
current_mode = None  # 'playback' or 'adquisicion'

def is_in_playback_window():
    now = datetime.now().time()
    return tstart_playback <= now <= tend_playback

def start_process(cmd):
    print(f"🚀 Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def stop_process(proc):
    if proc and proc.poll() is None:
        print("🛑 Stopping current process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("💣 Forcing kill...")
            proc.kill()

try:
    while True:
        in_playback = is_in_playback_window()

        if in_playback and current_mode != 'playback':
            stop_process(current_process)
            current_process = start_process(playback_cmd)
            current_mode = 'playback'

        elif not in_playback and current_mode != 'adquisicion':
            stop_process(current_process)
            current_process = start_process(acquisition_cmd)
            current_mode = 'adquisicion'

        time.sleep(1)  # Check every second

except KeyboardInterrupt: # Ctrl + C
    print("🔚 Interrupted. Cleaning up...")
    stop_process(current_process)
