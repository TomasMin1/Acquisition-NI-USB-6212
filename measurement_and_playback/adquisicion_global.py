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
import json

with open("Global code/config_global.json", "r") as f:
    config = json.load(f)

# -- Maximiza la prioridad del proceso --
def high_priority():
    try:
        p = psutil.Process(os.getpid())
        p.nice(-20)
    except Exception:
        pass

# --- Configuración ---
fs = config["fs"]
chunk_duration = config["chunk_duration"]
chunk_samples = int(chunk_duration * fs)
T_total = config["T_total"]
threshold = config["threshold"]
channels = config["channels"]
channel_names = config["channel_names"]
device = config["device"]
spectro_channel_idx = config["spectro_channel_idx"]
birdname = config["birdname"]
Route = config["Route"]

# -- Definiciones para poder utilizar los threads --
stop_event = threading.Event()
data_queue = queue.Queue()
save_queue = queue.Queue()
#Route = r'C:\Users\lsd\Documents\Codigo Adquisicion DAQ Git\Acquisition-NI-USB-6212' # Elegir donde se guarda
base_dir = birdname
today_str = time.strftime('%d-%m-%Y')

output_dir = os.path.join(Route, base_dir, today_str)

os.makedirs(output_dir, exist_ok=True)



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
        
        try:
            while i < total_chunks:
                reader.read_many_sample(
                    buffer,
                    number_of_samples_per_channel=chunk_samples,
                    timeout=chunk_duration*2
                )

                if is_above_threshold(buffer, spectro_channel_idx, threshold):
                    data_queue.put((i, buffer.copy()))
                    save_queue.put((i, buffer.copy()))
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ Saved")
                    i += 1
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ❌ Below threshold — skipped")

                target_time = start_time + (i + 1) * chunk_duration
                while True:
                    now = time.perf_counter()
                    if now >= target_time:
                        break
                    elif target_time - now > 0.005:
                        time.sleep(0.001)  # Coarse wait
                    else:
                        pass  # Spin-wait for sub-ms

        except Exception as e:
            print(f"❌ Error during acquisition: {e}")
                

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
    win.close()
    QApplication.quit()

from scipy.io.wavfile import write

import csv

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
                #scaled_wav = (scaled * 32767).astype(np.int32)
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




# -------- MAIN --------
'''Corre el codigo con las prioridades de hilos tal que se adquiera lo
más efectivamente posible'''

global_start = time.perf_counter()

t1 = threading.Thread(target=acquisition_thread, daemon=True)
t2 = threading.Thread(target=save_thread, daemon=True)
#t3 = threading.Thread(target= plotting_thread, daemon = True)

t1.start()
t2.start()

plotting_thread()

#t1.join()
#t2.join()

global_end = time.perf_counter()
print(f"\n⏱️ Total runtime: {global_end - global_start:.2f} seconds")
