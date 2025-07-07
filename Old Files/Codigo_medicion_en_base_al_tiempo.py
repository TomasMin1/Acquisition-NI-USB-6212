import time
import subprocess
from datetime import datetime, time as dtime

# Definí la ventana horaria para el playback
tstart_playback = dtime(11, 18, 0)  # 20:00
tend_playback   = dtime(11, 21, 0)  # 22:00

# Duraciones en segundos
session_silence = 30

def is_in_playback_window():
    now = datetime.now().time()
    return tstart_playback <= now <= tend_playback

while True:
    if is_in_playback_window():
        print("🎵 Ejecutando sesión con playback...")
        subprocess.run(['python', 'Adquisicion_guardado_playback.py'])

        # Después del playback, ejecutar sesión sin playback (en vez de silencio)
        print("🎤 Ejecutando sesión sin playback...")
        start = time.time()
        while time.time() - start < session_silence:
            subprocess.run(['python', 'Adquisicion y guardado.py'])
    else:
        print("⏰ Fuera de ventana de playback. Ejecutando sin playback continuamente...")
        subprocess.run(['python', 'Adquisicion y guardado.py'])
