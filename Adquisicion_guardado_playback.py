# Main acquisition code (no GUI)

import threading
import numpy as np
import queue
import time
import pyqtgraph as pg
#from pyqtgraph.Qt import QtGui
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
chunk_duration = 20 # Duración de cada wav
chunk_samples = int(chunk_duration * fs)
T_total = 60 # Tiempo total de medición
threshold = 0.01 # Valor de amplitud para usar de trigger
channels = ["ai0",'ai1'] # Canales que se están midiendo
channel_names = ["sound",'pressure'] # Nombre que aparece en el archivo wav correspondiente (Respetar orden de channels)
device = "Dev1" # NO MODIFICAR
spectro_channel_idx = 0 # Sobre qué canal plotear (orden en que son nombrados en channels arrancando de 0)
birdname = 'Tweetie' # Bird ID

# -- Definiciones para el protocolo de playback -- 
enable_playback = True
playback_folder = r'C:\Users\lsd\Desktop\tomisebalabo6\Tweetie\Playback'  # Carpeta con audios a reproducir
time_init_playback = dtime(9, 0)   # 09:00 AM
time_end_playback  = dtime(17, 0)  # 05:00 PM
playback_repeats = 1  # N_v: numero de veces que cada audio se reproduce
silence_duration = 5  # segundos de silencio entre audios

# -- Definiciones para poder utilizar los threads --
stop_event = threading.Event()
data_queue = queue.Queue()
save_queue = queue.Queue()
Route = r'C:\Users\lsd\Documents\Codigo Adquisicion DAQ Git\Acquisition-NI-USB-6212' # Elegir donde se guarda
base_dir = birdname
today_str = time.strftime('%d-%m-%Y')

output_dir = os.path.join(Route, base_dir, today_str)

os.makedirs(output_dir, exist_ok=True)


def is_within_playback_time_window(start, end):
    now = datetime.now().time()
    return start <= now <= end



# @njit
# def is_above_threshold(data, channel_idx, thresh_rms, thresh_zcr):
#     x = data[channel_idx][::10] # Toma los datos (toma cada 10 del original)
#     rms = np.sqrt(np.mean(x ** 2)) # Calcula RMS
#     zero_crossings = np.sum((x[1:] * x[:-1]) < 0) # Multiplica datos consecutivos y si da menos a 0 lo suma
#     zcr = zero_crossings / len(x) # Normaliza
#     return rms > thresh_rms and zcr > thresh_zcr


def is_above_threshold(data, channel_idx, thresh):
    x = data[channel_idx][::10] # Toma los datos (toma cada 10 del original)
    x -= np.mean(x)
    x = np.abs(x)
    value = np.quantile(x, .8)
    print("Trigger value obtained: ", value)
    return value > thresh



# -- Defino el buffer afuera del thread para poder optimizarlo con numba --
@njit
def define_buffer(n_channels, chunk_samples):
    return np.zeros((n_channels, chunk_samples))


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

            if is_above_threshold(buffer, spectro_channel_idx, threshold):
                data_queue.put((i, buffer.copy()))
                save_queue.put((i, buffer.copy()))
                #print(f"[{time.strftime('%H:%M:%S')}] ✅ Saved chunk {i}")
                i += 1
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ Below threshold — skipped")

            target_time = start_time + (chunk_idx + 1) * chunk_duration
            while True:
                now = time.perf_counter()
                if now >= target_time:
                    break
                elif target_time - now > 0.005:
                    time.sleep(0.001)  # Coarse wait
                else:
                    pass  # Spin-wait for sub-ms

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

    plot_waveform = win.addPlot(title=f"{channel_names[spectro_channel_idx]} waveform")
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
    time.sleep(chunk_duration)
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

            # Save audio and prepare stats
            timestamp = time.strftime('%d-%m-%Y_%H.%M.%S')
            for n in range(len(channels)):
                # Save per-channel WAV
                wav_filename = os.path.join(
                    output_dir,
                    f"{channel_names[n]}_{birdname}_{timestamp}.wav"
                )
                write(wav_filename, fs, scaled_wav[:, n])

                # Prepare CSV columns
                headers.append(f"{channel_names[n]}_avg")
                headers.append(f"{channel_names[n]}_ampl")
                row_stats.extend([avg_all[n], ampl_final[n]])

            # Save stats to a new CSV file for this measurement
            csv_filename = os.path.join(output_dir, f"{birdname}_{timestamp}.csv")
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerow(row_stats)

        except queue.Empty:
            time.sleep(0.1)



# def playback_thread():
#     print("🔊 Playback thread started.")
#     pygame.mixer.init()

#     # Get all .wav files in the folder
#     wav_files = glob.glob(os.path.join(playback_folder, "*.wav"))
#     if not wav_files:
#         print("⚠️ No .wav files found in playback folder.")
#         return

#     # Repeat playback N_v times
#     all_playbacks = []

#     for n in range(len(wav_files)):
#         i = 0
#         wav_files_new = []
#         while i < playback_repeats:
#             wav_files_new.append(wav_files[n])
#             i += 1
#         all_playbacks.append(wav_files_new)

#     random.shuffle(all_playbacks)

#     for filepath in all_playbacks:
#         if stop_event.is_set():
#             break

#         print(f"🎵 Playing: {os.path.basename(filepath)}")
#         pygame.mixer.music.load(filepath)
#         pygame.mixer.music.play()

#         # Wait until it finishes
#         while pygame.mixer.music.get_busy():
#             time.sleep(0.1)

#         # Silence (pause)
#         time.sleep(silence_duration)

#     print("🔇 Playback thread finished.")


def playback_thread():
    print("🔊 Playback thread started.")
    pygame.mixer.init()

    # Get all .wav files in the folder
    wav_files = glob.glob(os.path.join(playback_folder, "*.wav"))
    if not wav_files:
        print("⚠️ No .wav files found in playback folder.")
        return

    session_count = 0
    while is_within_playback_time_window(time_init_playback, time_end_playback):
        # Create one session
        base_playbacks = [[wav] * playback_repeats for wav in wav_files]
        random.shuffle(base_playbacks)

        session_count += 1
        print(f"\n--- Session {session_count} ---")
        for block in base_playbacks:
            for wav in block:
                if stop_event.is_set():
                    break

                print(f"🎵 Playing: {os.path.basename(wav)}")
                pygame.mixer.music.load(wav)
                pygame.mixer.music.play()

                # Wait until it finishes
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                # Silence (pause)
                time.sleep(silence_duration)
                print(f"Playing {wav}")
                # Insert playback code here, or simulate:
                time.sleep(0.1)  # simulate playback delay

        # Optional pause between sessions
        time.sleep(1)  # or more, depending on your setup


    print("🔇 Playback thread finished.")


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
