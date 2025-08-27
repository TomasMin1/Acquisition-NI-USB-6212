import time
import subprocess
from datetime import datetime, time as dtime
import json


#-------------------------- Configuracion para medicion sin playback -------------------------
config = {
    "fs": 44150,  # Frecuencia de sampleo
    "chunk_duration": 60,  # Duración de cada wav
    "chunk_samples": int(30 * 44150), #numero de la derecha es frec. de sampleo
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
    "Route": r"C:\Users\lsd\Documents\Acquisition-NI-USB-6212-main\measurement_and_playback"
}

# Ruta del archivo de configuración
config_path = r"C:\Users\lsd\Documents\Acquisition-NI-USB-6212-main\measurement_and_playback\config_global.json"

# Guardar archivo JSON
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"✅ Configuración guardada en: {config_path}")


#-------------------------- Configuracion para medicion con playback -------------------------
config_regimen_con_playback = config.copy()
config_regimen_con_playback["T_total"] = 120  # e.g. 30 seconds for short sessions
config_regimen_con_playback["chunk_duration"] = 30
config_regimen_con_playback["chunk_samples"] = int(30 * config["fs"])
config_regimen_con_playback_path = r"C:\Users\lsd\Documents\Acquisition-NI-USB-6212-main\measurement_and_playback\config_regimen_con_playback.json"

with open(config_regimen_con_playback_path, 'w') as f:
    json.dump(config_regimen_con_playback, f, indent=4)

print(f"✅ Configuración (régimen con playback) guardada en: {config_regimen_con_playback_path}")


#-------------------------- Configuracion horarios con playback -------------------------
tstart_playback = dtime(13, 16, 0)
tend_playback   = dtime(13, 20, 0)


#---------------------------------- Codigo ----------------------------------------------
playback_cmd_directory = r"C:\Users\lsd\Documents\Acquisition-NI-USB-6212-main\measurement_and_playback\playback_global.py"
aquisition_cmd_directory = r"C:\Users\lsd\Documents\Acquisition-NI-USB-6212-main\measurement_and_playback\adquisicion_global.py"
aquisition_during_playback_cmd_directory = r"C:\Users\lsd\Documents\Acquisition-NI-USB-6212-main\measurement_and_playback\adquisicion_durante_playback.py"

# Comandos
playback_cmd = ['python', playback_cmd_directory, config_regimen_con_playback_path]
acquisition_cmd = ['python', aquisition_cmd_directory, config_path]
acquisition_during_playback_cmd = ['python', aquisition_during_playback_cmd_directory, config_regimen_con_playback_path]

# Estado actual
current_process = None
current_mode = None  # 'playback' or 'adquisicion'

def is_in_playback_window():
    now = datetime.now().time()
    return tstart_playback <= now <= tend_playback

def start_process(cmd):
    print(f"Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

def stop_process(proc):
    if proc and proc.poll() is None:
        print("Stopping current process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Forcing kill...")
            proc.kill()

try:
    current_process = None
    current_mode = None

    while True:
        in_playback = is_in_playback_window()

        if in_playback:
            if current_mode != 'playback_loop':
                stop_process(current_process)
                current_process = None
                current_mode = 'playback_loop'
                print("Entrando en modo de playback alternado.")

            # Alternado entre playback y no playback
            print("Ejecutando sesión con playback...")
            subprocess.run(playback_cmd)

            print("Ejecutando sesión sin playback...")
            subprocess.run(acquisition_during_playback_cmd)

        else:
            if current_mode != 'adquisicion':
                stop_process(current_process)
                print("Ejecutando adquisición continua sin playback...")
                current_process = start_process(acquisition_cmd)
                current_mode = 'adquisicion'

            time.sleep(1)

except KeyboardInterrupt: # Ctrl + C
    print("🛑 Interrumpido por el usuario. Finalizando...")
    stop_process(current_process)