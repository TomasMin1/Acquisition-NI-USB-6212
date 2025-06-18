#Main acquisition code (no GUI)

import threading
import numpy as np
import queue
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from scipy.signal import spectrogram
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
from PyQt5.QtGui import QTransform
from numba import njit
import pandas as pd
import os
import psutil
import gc
from scipy.io.wavfile import write


# Prioritize process
def high_priority():
    try:
        p = psutil.Process(os.getpid())
        p.nice(-20)
        #print('✅ Process priority raised to highest')
    except Exception:
        pass

# Configuration
fs = 44150
chunk_duration = 5
chunk_samples = int(chunk_duration * fs)
T_total = 20
threshold = 0.0
channels = ["ai0",'ai1']
channel_names = ["sound",'pressure']
device = "Dev1"
spectro_channel_idx = 0
birdname = 'Tweetie'

stop_event = threading.Event()
data_queue = queue.Queue()
save_queue = queue.Queue()
output_dir = birdname
os.makedirs(output_dir, exist_ok=True)

@njit
def is_above_threshold(data, channel_idx, thresh):
    return np.max(data[channel_idx][::100]) > thresh

@njit
def define_buffer(n_channels, chunk_samples):
    return np.zeros((n_channels, chunk_samples))

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

            #next_chunk_time = start_time + (chunk_idx + 1) * chunk_duration
            #sleep_time = next_chunk_time - time.perf_counter()
            #if sleep_time > 0:
            #    time.sleep(sleep_time)
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

def plotting_thread():
    app = QtGui.QApplication.instance() or QtGui.QApplication([])

    win = pg.GraphicsLayoutWidget(title="DAQ Live Viewer")
    win.resize(1000, 600)

    plot_waveform = win.addPlot(title="Waveform")
    curve_waveform = plot_waveform.plot(pen='y')

    win.nextRow()
    plot_spectrogram = win.addPlot(title="Spectrogram")
    img = pg.ImageItem()
    plot_spectrogram.addItem(img)
    win.show()

    while not stop_event.is_set() or not data_queue.empty():
        try:
            i, data = data_queue.get_nowait()
        except queue.Empty:
            QtGui.QApplication.processEvents()
            continue

        # Waveform
        y = data[spectro_channel_idx][::10]
        x = np.linspace(0, chunk_duration, len(y))
        curve_waveform.setData(x[:200], y[:200])

        # Spectrogram
        f_spec, t_spec, Sxx = spectrogram(data[spectro_channel_idx][::5], fs=fs/5, nperseg=256, noverlap=128)
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)

        img.setImage(Sxx_dB.T, levels=(Sxx_dB.max(), Sxx_dB.min()))
        img.resetTransform()
        dx = t_spec[1] - t_spec[0]
        dy = f_spec[1] - f_spec[0]
        img.setTransform(QTransform().scale(dx, dy))
        img.setPos(0, 0)

        QtGui.QApplication.processEvents()

    print("🛑 Plotting thread finished.")
    win.close()
    QtGui.QApplication.quit()

from scipy.io.wavfile import write

import csv

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
                scaled = centered / ampl_all                   # normalize per channel

            # Save audio and prepare stats
            timestamp = time.strftime('%d-%m-%Y_%H.%M.%S')
            for n in range(len(channels)):
                # Save per-channel WAV
                wav_filename = os.path.join(
                    output_dir,
                    f"{channel_names[n]}_{birdname}_{timestamp}.wav"
                )
                write(wav_filename, fs, scaled[:, n])

                # Prepare CSV columns
                headers.append(f"{channel_names[n]}_avg")
                headers.append(f"{channel_names[n]}_ampl")
                row_stats.extend([avg_all[n], ampl_all[n]])

            # Save stats to a new CSV file for this measurement
            csv_filename = os.path.join(output_dir, f"{birdname}_{timestamp}.csv")
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerow(row_stats)

        except queue.Empty:
            time.sleep(0.1)




# -------- MAIN --------
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
