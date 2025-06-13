import threading
import numpy as np
import queue
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from scipy.signal import spectrogram
import pandas as pd
import os
from numba import njit
from PyQt5.QtGui import QTransform

# --- Configuration ---
fs = 44150
chunk_duration = 3               # seconds
chunk_samples = int(chunk_duration * fs)
threshold = 0.5
channels = ["ai0", "ai1"]
spectro_channel_idx = 0
T_total = 10                     # total acquisition time in seconds
output_dir = "simulated_chunks"
os.makedirs(output_dir, exist_ok=True)

# Shared data queue
data_queue = queue.Queue()
stop_event = threading.Event()  # used to stop acquisition after T_total

# --- Simulate data ---
def generate_fake_data(num_channels, num_samples, fs):
    t = np.linspace(0, num_samples / fs, num_samples, endpoint=False)
    data = np.zeros((num_channels, num_samples))
    for ch in range(num_channels):
        freq = 440 + ch * 100
        signal = 0.6 * np.sin(2 * np.pi * freq * t)
        noise = 0.1 * np.random.randn(num_samples)
        data[ch] = signal + noise
    return data

# --- Threshold check compiled for speed ---
@njit
def is_above_threshold(data, channel_idx, thresh):
    return np.max(data[channel_idx]) > thresh

# --- Acquisition thread (simulated) ---
def acquisition_thread():
    i = 0
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        if elapsed >= T_total:
            stop_event.set()
            break

        time.sleep(chunk_duration)  # Simulate DAQ delay
        data = generate_fake_data(len(channels), chunk_samples, fs)

        if is_above_threshold(data, spectro_channel_idx, threshold):
            print(f"[{time.strftime('%H:%M:%S')}] Data above threshold — saving chunk {i}")
            # Save to disk
            df = pd.DataFrame(data.T, columns=channels)
            df.to_pickle(os.path.join(output_dir, f"chunk_{i:04d}.pkl"))
            i += 1

            # Send to plotting queue
            data_queue.put(data.copy())
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Data below threshold — skipping")

# --- Plotting thread ---
def plotting_thread():
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    # PyQtGraph setup
    win = pg.GraphicsLayoutWidget(title="Simulated DAQ Viewer")
    win.resize(1000, 800)

    # Waveform
    plot_waveform = win.addPlot(title="Waveform")
    curve_waveform = plot_waveform.plot(pen='y')

    # Spectrogram
    win.nextRow()
    plot_spectrogram = win.addPlot(title="Spectrogram")
    img = pg.ImageItem()
    plot_spectrogram.addItem(img)

    win.show()

    while not stop_event.is_set() or not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
        except queue.Empty:
            continue

        y = data[spectro_channel_idx][::10]
        x = np.linspace(0, chunk_duration, len(y))
        curve_waveform.setData(x, y)

        f_spec, t_spec, Sxx = spectrogram(data[spectro_channel_idx], fs=fs, nperseg=1024, noverlap=512)
        Sxx_dB = 10 * np.log10(Sxx + 1e-12)

        img.setImage(Sxx_dB.T, levels=(Sxx_dB.min(), Sxx_dB.max()))
        img.resetTransform()
        dx = t_spec[1] - t_spec[0]
        dy = f_spec[1] - f_spec[0]
        transform = QTransform().scale(dx, dy)
        img.setTransform(transform)

        img.setPos(0, 0)

        QtWidgets.QApplication.processEvents()

    print("Plotting thread finished.")
    win.close()
    QtWidgets.QApplication.quit()

# --- Run threads ---
if __name__ == "__main__":
    print("🔁 Starting simulated DAQ system...")
    acq_thread = threading.Thread(target=acquisition_thread, daemon=True)
    acq_thread.start()

    plotting_thread()  # blocks until UI is closed
    print("✅ Simulation complete.")
