# Código con protocolo playback

import threading
import numpy as np
import queue
import time
import pyqtgraph as pg
from scipy.signal import spectrogram
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication
from pyqtgraph.Qt import QtWidgets
from numba import njit
import pandas as pd
import os
import psutil
import gc
from scipy.io.wavfile import write
import pygame
import random
import glob
from datetime import datetime, time as dtime
from scipy.io.wavfile import write
import csv

# -- Maximiza la prioridad del proceso --
def high_priority():
    try:
        p = psutil.Process(os.getpid())
        p.nice(-20)
    except Exception:
        pass

# --- Configuración ---
fs = 44150 # Frecuencia de sampleo
chunk_duration = 30 # Duración de cada wav
chunk_samples = int(chunk_duration * fs)
T_total = 100000 # Tiempo total de medición
threshold = 0.01 # Valor de amplitud para usar de trigger
channels = ["ai0",'ai1'] # Canales que se están midiendo
channel_names = ["sound",'pressure'] # Nombre que aparece en el archivo wav correspondiente (Respetar orden de channels)
device = "Dev2" # NO MODIFICAR
spectro_channel_idx = 0 # Sobre qué canal plotear (orden en que son nombrados en channels arrancando de 0)
birdname = 'Tweetie' # Bird ID

# -- Definiciones para el protocolo de playback -- 
enable_playback = True
playback_folder = r'C:\Users\lsd\Desktop\tomisebalabo6\Tweetie\Playback'  # Carpeta con audios a reproducir
playback_repeats = 7  # N_v: numero de veces que cada audio se reproduce
#silence_duration = 5  # segundos de silencio entre audios
Route = r'C:\Users\lsd\Documents\Codigo Adquisicion DAQ Git\Acquisition-NI-USB-6212' # Elegir donde se guarda

# -- Definiciones para poder utilizar los threads --
stop_event = threading.Event()
data_queue = queue.Queue()
save_queue = queue.Queue()
playback_queue = queue.Queue()
trigger_queue = queue.Queue()
Route = r'C:\Users\lsd\Documents\Codigo Adquisicion DAQ Git\Acquisition-NI-USB-6212' # Elegir donde se guarda
base_dir = birdname
today_str = time.strftime('%d-%m-%Y')

output_dir = os.path.join(Route, base_dir, today_str)

os.makedirs(output_dir, exist_ok=True)


def is_within_playback_time_window(start, end):
    now = datetime.now().time()
    return start <= now <= end


# -- Defino el buffer afuera del thread para poder optimizarlo con numba --
@njit
def define_buffer(n_channels, chunk_samples):
    return np.zeros((n_channels, chunk_samples))


def playback_thread():
    print("🔊 Playback thread started.")
    pygame.mixer.init()

    # Get all .wav files in the folder
    wav_files = glob.glob(os.path.join(playback_folder, "*.wav"))
    
    if not wav_files:
        print("⚠️ No .wav files found in playback folder.")
        return

    session_count = 0
    session_wavs = []

    while True:
        # If session exhausted, start a new one
        if not session_wavs:
            session_count += 1
            #print(f"\n--- Session {session_count} ---")
            session_wavs = []
            for wav in wav_files:
                session_wavs.extend([wav] * playback_repeats)

            random.shuffle(session_wavs)

        try:
            time.sleep(0.5)
            # Wait for acquisition to trigger playback
            trigger = trigger_queue.get(timeout=1)
            if trigger == 1:
                wav_to_play = session_wavs.pop(0)
                print(f"🎵 Playing {os.path.basename(wav_to_play)}")
                pygame.mixer.music.load(wav_to_play)
                pygame.mixer.music.play()
                playback_queue.put(os.path.basename(wav_to_play))
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except queue.Empty:
            continue  # Just loop again if no trigger in 1s

'''Toma los datos del NIDAQ para los canales indicados y 
si superan el threshold los guarda en data_queue y save_queue
para que los siguientes hilos trabajen con esos datos. Una vez que 
guarda esos datos continua con la adquisición del próximo chunk'''
def acquisition_thread():

    high_priority()
    gc.collect()
    gc.disable()

    total_samples = int(fs * T_total)
    total_chunks = total_samples // chunk_samples
    i = 0
    buffer = define_buffer(len(channels), chunk_samples)
    start_time = time.perf_counter()

    trigger_queue.put(1)
    
    #print(f"🎙️  Starting acquisition for {T_total} seconds ({total_chunks} chunks of {chunk_duration}s)...")

    with nidaqmx.Task() as task:
        for ch in channels:
            task.ai_channels.add_ai_voltage_chan(f"{device}/{ch}")

        task.timing.cfg_samp_clk_timing(
            rate=fs,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=chunk_samples
        )

        reader = AnalogMultiChannelReader(task.in_stream)
        task.start()

        for chunk_idx in range(total_chunks):
            reader.read_many_sample(
                buffer,
                number_of_samples_per_channel=chunk_samples,
                timeout=chunk_duration*2
            )

            # 🟢 Trigger playback for each new acquisition
            trigger_queue.put(1)

            # Timing logic (to match precise acquisition intervals)
            target_time = start_time + (chunk_idx + 1) * chunk_duration
            while True:
                now = time.perf_counter()
                if now >= target_time:
                    break
                elif target_time - now > 0.005:
                    time.sleep(0.001)
                else:
                    pass

            # Send data to queues
            data_queue.put((i, buffer.copy()))
            save_queue.put((i, buffer.copy()))
            i += 1

            print(f"[{time.strftime('%H:%M:%S')}] ✅ Saved")

            # Repeat wait block (if needed)
            target_time = start_time + (chunk_idx + 1) * chunk_duration
            while True:
                now = time.perf_counter()
                if now >= target_time:
                    break
                elif target_time - now > 0.005:
                    time.sleep(0.001)
                else:
                    pass


    stop_event.set()
    elapsed_total = time.perf_counter() - start_time
    print(f"\n🟩 Acquisition thread finished. Chunks acquired: {i}. Took {elapsed_total:.2f}s")

    gc.enable()


'''Revisa data_queue y, si tiene datos, los plotea (solo el canal indicado). 
Hace un plot de f(x) para 400 puntos centrado en el máximo absoluto
y un plot del espectograma de todo el chunk.'''

def plotting_thread():
    #app = QtGui.QApplication.instance() or QtGui.QApplication([])
    app = QApplication.instance() or QApplication([])

    win = pg.GraphicsLayoutWidget(title="DAQ Live Viewer")
    win.resize(1000, 600)

    plot_waveform = win.addPlot(title=f"{channel_names[spectro_channel_idx]} waveform - {time.strftime('%H:%M:%S')}")
    curve_waveform = plot_waveform.plot(pen='y')

    win.nextRow()
    plot_spectrogram = win.addPlot(title=f"{channel_names[spectro_channel_idx]} spectrogram")
    img = pg.ImageItem()
    plot_spectrogram.addItem(img)
    win.show()

    while not stop_event.is_set() or not data_queue.empty():
        try:
            i, data = data_queue.get_nowait()
        except queue.Empty:
            #QtGui.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()
            continue

        # --- Waveform ---
        plot_waveform.setTitle(f"{channel_names[spectro_channel_idx]} waveform - {time.strftime('%H:%M:%S')}")
        y = data[spectro_channel_idx][::10]
        x = np.linspace(0, chunk_duration, len(y))

        # cent = np.where(abs(y) == max(abs(y)))[0][0]
        # disp = 200
        #curve_waveform.setData(x[cent-disp:cent+disp], y[cent-disp:cent+disp])
        curve_waveform.setData(x, y)

        # --- Spectrogram ---
        f_spec, t_spec, Sxx = spectrogram(data[spectro_channel_idx][::2], fs=fs/2, nperseg=256, noverlap=128)
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)

        img.setImage(Sxx_dB.T, levels=(Sxx_dB.max(), Sxx_dB.min()))
        img.resetTransform()
        dx = t_spec[1] - t_spec[0]
        dy = f_spec[1] - f_spec[0]
        img.setTransform(QTransform().scale(dx, dy))
        img.setPos(0, 0)

        #QtGui.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
        
    print("🛑 Plotting thread finished.")
    time.sleep(5)
    win.close()
    QApplication.quit()


'''Revisa save_thread y, si tiene datos, los guarda en formato wav (uno por cada canal). 
Además calcula el promedio y valor máximo de cada chunk y los guarda en un csv para 
poder obtener la señal inversa (de vuelta indicando de qué canal se trata). '''

def save_thread():
    while not stop_event.is_set() or not save_queue.empty():
        try:
            i, data = save_queue.get_nowait()
            data = data.T  # shape: (samples, channels)

            # Initialize row storage
            row_stats = []
            headers = []

            max_val = np.max(np.abs(data))
            if max_val == 0:
                scaled = np.zeros_like(data, dtype=np.int16)
                avg_all = np.zeros(data.shape[1])
                ampl_all = np.zeros(data.shape[1])
            else:
                avg_all = np.mean(data, axis=0)                # shape: (channels,)
                centered = data - avg_all                      # center each channel
                ampl_all = np.max(np.abs(centered), axis=0)    # shape: (channels,)
                scaled = centered / ampl_all
                scaled_wav = (scaled * 32767).astype(np.int16) # normalize per channel
                ampl_final = ampl_all/32767

            playback_name = playback_queue.get_nowait()

            # Save audio and prepare stats
            timestamp = time.strftime('%d-%m-%Y_%H.%M.%S')
            for n in range(len(channels)):
                # Save per-channel WAV
                wav_filename = os.path.join(
                    output_dir,
                    f"Playback({playback_name})_{channel_names[n]}_{birdname}_{timestamp}.wav"
                )
                write(wav_filename, fs, scaled_wav[:, n])

                # Prepare CSV columns
                headers.append(f"{channel_names[n]}_avg")
                headers.append(f"{channel_names[n]}_ampl")
                row_stats.extend([avg_all[n], ampl_final[n]])

            # Save stats to a new CSV file for this measurement
            csv_filename = os.path.join(output_dir, f"Playback({playback_name})_{birdname}_{timestamp}.csv")
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerow(row_stats)

        except queue.Empty:
            time.sleep(0.1)





# -------- MAIN --------
'''Corre el codigo con las prioridades de hilos tal que se adquiera lo
más efectivamente posible'''

global_start = time.perf_counter()

t1 = threading.Thread(target=acquisition_thread, daemon=True)
t2 = threading.Thread(target=save_thread, daemon=True)
threads = [t1, t2]

if enable_playback:
    t3 = threading.Thread(target=playback_thread, daemon=True)
    threads.append(t3)

for t in threads:
    t.start()


plotting_thread()

global_end = time.perf_counter()
print(f"\n⏱️ Total runtime: {global_end - global_start:.2f} seconds")