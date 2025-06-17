#Main acquisition code

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

 # Prioridades
try:
    psutil.Process().nice(-10)
    print('Process priority raised to highest')

except Exception as e:
    print ('')

# Configuration
fs = 44150
chunk_duration = 5
chunk_samples = int(chunk_duration * fs)
T_total = 20  # seconds
threshold = 0.5
channels = ["ai0", "ai1"] #channels to acquire
device = "Dev1"
spectro_channel_idx = 0

stop_event = threading.Event()

data_queue = queue.Queue()
save_queue = queue.Queue()
output_dir = "daq_data"
os.makedirs(output_dir, exist_ok=True)

@njit
def is_above_threshold(data, channel_idx, thresh):
    return np.max(data[channel_idx]) > thresh

def run_acquisition(fs, chunk_duration, threshold, T_total, output_dir, channels, spectro_channel_idx):
    chunk_samples = int(chunk_duration * fs)

    stop_event = threading.Event()
    data_queue = queue.Queue()
    os.makedirs(output_dir, exist_ok=True)
    def acquisition_thread():
        print(f"Starting acquisition for {T_total} seconds (chunk size = {chunk_duration}s)...")

        total_samples = int(fs * T_total)
        total_chunks = total_samples // chunk_samples
        i = 0

        buffer = np.zeros((len(channels), chunk_samples))
        start_time = time.time()

        with nidaqmx.Task() as task:
            # Configure channels
            for ch in channels:
                task.ai_channels.add_ai_voltage_chan(f"{device}/{ch}")
            
            # Configure timing
            task.timing.cfg_samp_clk_timing(
                rate=fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=chunk_samples
            )

            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()

            for chunk_idx in range(total_chunks):
                chunk_start = time.time()

                # Read one chunk of data
                reader.read_many_sample(
                    buffer,
                    number_of_samples_per_channel=chunk_samples,
                    timeout=10.0
                )

                if is_above_threshold(buffer, spectro_channel_idx, threshold):
                    data_queue.put((i, buffer.copy()))
                    save_queue.put((i, buffer.copy()))
                    print(f"[{time.strftime('%H:%M:%S')}] Data above threshold — saved chunk {i}")
                    i += 1
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Data below threshold — skipped")

                # Maintain real-time pacing
                #elapsed = time.time() - chunk_start
                #remaining = chunk_duration - elapsed
                next_chunk_time = start_time + (chunk_idx + 1) * chunk_duration
                now = time.time()
                sleep_time = next_chunk_time - now

                if sleep_time > 0:
                    time.sleep(sleep_time)

        stop_event.set()
        elapsed_total = time.time() - start_time
        print(f"\n✅ Acquisition completed in {elapsed_total:.2f} seconds. Total chunks processed: {i}")

    def plotting_thread():
        if not QtGui.QApplication.instance():
            app = QtGui.QApplication([])
        else:
            app = QtGui.QApplication.instance()

        win = pg.GraphicsLayoutWidget(title="DAQ Live Viewer")
        win.resize(1000, 800)

        plot_waveform = win.addPlot(title="Waveform")
        curve_waveform = plot_waveform.plot(pen='y')

        win.nextRow()
        plot_spectrogram = win.addPlot(title="Spectrogram")
        img = pg.ImageItem()
        plot_spectrogram.addItem(img)
        win.show()

        while not stop_event.is_set() or not data_queue.empty():
            try:
                i, data = data_queue.get(timeout=1)
            except queue.Empty:
                continue

            # plot waveform
            y = data[spectro_channel_idx][::10]
            x = np.linspace(0, chunk_duration, len(y))
            curve_waveform.setData(x, y)

            # spectrogram
            f_spec, t_spec, Sxx = spectrogram(data[spectro_channel_idx], fs=fs, nperseg=128, noverlap=64)
            Sxx_dB = 10 * np.log10(Sxx + 1e-12)

            #img.setImage(Sxx_dB.T, levels=(Sxx_dB.min(), Sxx_dB.max()))
            #img.resetTransform()
            #img.scale(t_spec[1] - t_spec[0], f_spec[1] - f_spec[0])
            #img.setPos(0, 0)
            img.setImage(Sxx_dB.T, levels=(Sxx_dB.min(), Sxx_dB.max()))
            img.resetTransform()
            dx = t_spec[1] - t_spec[0]
            dy = f_spec[1] - f_spec[0]
            transform = QTransform().scale(dx, dy)
            img.setTransform(transform)

            img.setPos(0, 0)


            QtGui.QApplication.processEvents()

        print("🛑 Plotting thread finished.")
        win.close()
        QtGui.QApplication.quit()

    def save_thread():
        while not stop_event.is_set() or not save_queue.empty():
            try:
                i, data = save_queue.get(timeout=1)
                df = pd.DataFrame(data.T, columns=channels)
                df.to_pickle(os.path.join(output_dir, f"chunk_{i:04d}.pkl"))
                # np.save(os.path.join(output_dir,f'chunk_{i:04d}.npy'), data.T)
            except queue.Empty:
                continue
    
    t1 = threading.Thread(target=acquisition_thread, daemon=True)
    t2 = threading.Thread(target = save_thread, daemon = True)
    t2.start()
    t1.start()

    plotting_thread()
'''
def start_daq_system(fs, chunk_duration, T_total, threshold, channels, output_dir, spectro_channel_name):
    # Recalculate needed variables
    chunk_samples = int(chunk_duration * fs)
    total_samples = int(fs * T_total)
    total_chunks = total_samples // chunk_samples
    device = "Dev1"  # or make this configurable too

    spectro_channel_idx = channels.index(spectro_channel_name)
    stop_event.clear()

    # Update global config
    acquisition_params = {
        'fs': fs,
        'chunk_duration': chunk_duration,
        'T_total': T_total,
        'threshold': threshold,
        'channels': channels,
        'output_dir': output_dir,
        'spectro_channel_idx': spectro_channel_idx,
        'chunk_samples': chunk_samples,
        'total_chunks': total_chunks,
        'device': device
    }

    # Start acquisition + plotting
    t1 = threading.Thread(target=acquisition_thread, args=(acquisition_params,))
    t1.start()

    plotting_thread(acquisition_params)


# Start threads
t1 = threading.Thread(target=acquisition_thread, daemon=True)
t2 = threading.Thread(target = save_thread, daemon = True)
t2.start()
t1.start()

plotting_thread()
'''


