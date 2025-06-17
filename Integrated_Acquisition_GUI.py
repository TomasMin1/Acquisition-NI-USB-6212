# acquisition with gui integration

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys, os, json
from Adquisicion_Guardado import run_acquisition  # Import acquisition logic from separate file
import os
from PyQt5.QtGui import QPixmap
import numpy as np
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
from PyQt5.QtCore import QTimer


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "daq_config.json")

class DAQConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(800, 600)
        self.setWindowTitle("Configuracion para la adquisicion de datos")
        self.setWindowIcon(QIcon("icon.png"))
        self.setStyleSheet("background-color: #2A2E32; color: white;")
        self.layout = QVBoxLayout()

        self.fs_input = self.add_input("Frecuencia de sampleo [fs]")
        self.chunk_duration_input = self.add_input("Duracion de los archivos guardados [s]")
        self.threshold_input = self.add_input("Threshold [V]")
        self.channels_input = self.add_input("Canales (separados por coma, ejemplo: ai0,ai1)")
        self.spectro_channel_idx_input = self.add_input("Canal sobre el que se quiere ver su espectrograma (ejemplo: ai0)")
        self.T_total_input = self.add_input("Tiempo de adquisicion total")

        self.add_output_dir_picker()

        self.run_button = QPushButton("Comenzar Adquisicion")
        self.run_button.setStyleSheet("background-color: #4990E2; color: white;")
        self.run_button.clicked.connect(self.on_run_clicked)
        self.layout.addWidget(self.run_button)

        self.setLayout(self.layout)
        self.load_previous_config()

        # graph
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.hide()
        self.layout.addWidget(self.canvas)
        
        self.plot_waveform = pg.PlotWidget(title="Waveform")
        self.plot_spectrogram = pg.PlotWidget(title="Spectrogram")
        self.img = pg.ImageItem()
        self.plot_spectrogram.addItem(self.img)
        
        self.plot_waveform.hide()
        self.plot_spectrogram.hide()
        
        self.layout.addWidget(self.plot_waveform)
        self.layout.addWidget(self.plot_spectrogram)
        

    def add_input(self, label_text):
        row = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit()
        input_field.setFixedWidth(200)
        input_field.setStyleSheet("background-color: #1E1E1E; color: white;")
        row.addWidget(label)
        row.addWidget(input_field, alignment=Qt.AlignRight)
        self.layout.addLayout(row)
        return input_field

    def add_output_dir_picker(self):
        row = QHBoxLayout()
        label = QLabel("Output Directory")
        self.output_dir_display = QLineEdit()
        self.output_dir_display.setReadOnly(True)
        self.output_dir_display.setStyleSheet("background-color: #1E1E1E; color: white;")

        choose_button = QPushButton("Choose...")
        choose_button.setStyleSheet("background-color: #4990E2; color: white;")
        choose_button.clicked.connect(self.choose_output_dir)

        row.addWidget(label)
        row.addWidget(self.output_dir_display)
        row.addWidget(choose_button)
        self.layout.addLayout(row)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_display.setText(directory)

    def load_previous_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            self.fs_input.setText(str(config.get("fs", "")))
            self.chunk_duration_input.setText(str(config.get("chunk_duration", "")))
            self.threshold_input.setText(str(config.get("threshold", "")))
            self.channels_input.setText(",".join(config.get("channels", [])))
            self.spectro_channel_idx_input.setText(config.get("spectro_channel", ""))
            self.T_total_input.setText(str(config.get("T_total", "")))
            self.output_dir_display.setText(config.get("output_dir", ""))

    def on_run_clicked(self):
        fs = int(self.fs_input.text())
        chunk_duration = float(self.chunk_duration_input.text())
        threshold = float(self.threshold_input.text())
        T_total = float(self.T_total_input.text())
        output_dir = self.output_dir_display.text()
        channels = [ch.strip() for ch in self.channels_input.text().split(',') if ch.strip()]
        spectro_channel = self.spectro_channel_idx_input.text().strip()

        if spectro_channel not in channels:
            print(f"❌ Error: canal '{spectro_channel}' no está en {channels}")
            return

        spectro_channel_idx = channels.index(spectro_channel)

        # Save config
        config = {
            "fs": fs,
            "chunk_duration": chunk_duration,
            "threshold": threshold,
            "T_total": T_total,
            "channels": channels,
            "spectro_channel": spectro_channel,
            "output_dir": output_dir
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)

        # ✅ Start acquisition logic here
        run_acquisition(fs, chunk_duration, threshold, T_total, 
        output_dir, channels, spectro_channel_idx)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    #  visual config. (nice)
    app.setStyleSheet("""
        QWidget {
            background-color: #2A2E32;
            color: white;
            font-size: 14px;
        }

        QLabel {
            color: white;
        }

        QLineEdit {
            background-color: #1E1E1E;
            color: white;
            border: 1px solid #555;
            padding: 4px;
        }

        QPushButton {
            background-color: #4990E2;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
        }

        QPushButton:hover {
            background-color: #5AA0F2;
        }

        QPushButton:pressed {
            background-color: #3A7AC2;
        }
    """)

    window = DAQConfigWindow()
    window.show()
    sys.exit(app.exec_())
