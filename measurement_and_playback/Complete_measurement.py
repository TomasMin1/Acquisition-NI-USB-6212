import time
import subprocess
from datetime import datetime, time as dtime
import json

config = {
    "fs": 44150,  # Frecuencia de sampleo
    "chunk_duration": 60,  # Duración de cada wav
    "chunk_samples": int(30 * 44150),
    "T_total": 100000,  # Tiempo total de medición (para que sea endless poner un numero mas grande que el regimen de tiempo en el que no tengo playback)
    "threshold": 0.01,  # Trigger
    "channels": ["ai0", "ai1"],
    "channel_names": ["sound", "pressure"],
    "device": "Dev1", # Ver en PC como se llama el DAQ, para el NI-USB-6212 suele ser Dev1
    "spectro_channel_idx": 0, # Plotea el primero de la lista
    "birdname": "Tweetie",

    # --- Playback ---
    "enable_playback": True,
    "playback_folder": r"C:\Users\lsd\Desktop\tomisebalabo6\Tweetie\Playback",
    "playback_repeats": 2,
    
    # Ruta donde se guarda todo
    "Route": r"C:\Users\lsd\Documents\Codigo Adquisicion DAQ Git\Acquisition-NI-USB-6212"
}

# Ruta del archivo de configuración
config_path = "measurement_and_playback/config_global.json"

# Guardar archivo JSON
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"✅ Configuración guardada en: {config_path}")

# Horarios para playback
tstart_playback = dtime(14, 55, 0)
tend_playback   = dtime(14, 56, 0)

# Comandos
playback_cmd = ['python', 'measurement_and_playback/playback_global.py']
acquisition_cmd = ['python', 'measurement_and_playback/adquisicion_global.py']

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
