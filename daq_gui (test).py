from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import sys
import json
import os

#file that saves previously used config (saves time next time you open the app)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daq_config.json")

class DAQConfigWindow(QWidget):
    def __init__(self):
        super().__init__() # initializes Qtwidget
        self.setFixedSize(800, 600)  # px #remove if you dont want the window to be a fixed size
        self.setWindowTitle("Configuracion para la adquisicion de datos")
        self.layout = QVBoxLayout()

        #inputs
        config = self.load_config()

        self.fs_input = self.add_input("Frecuencia de sampleo [fs]", config.get("fs", ""))
        self.chunk_duration_input = self.add_input("Duracion de los archivos guardados [s]", config.get("chunk_duration", ""))
        self.threshold_input = self.add_input("Threshold [V]", config.get("threshold", ""))
        self.channels_input = self.add_input("Canales (separados por coma, ejemplo: ai0,ai1)", config.get("channels", ""))
        self.spectro_channel_idx_input = self.add_input("Canal sobre el que se quiere ver su espectrograma (ejemplo: ai0))", config.get("spectro_channel", ""))
        self.T_total_input = self.add_input("Tiempo de adquisicion total", config.get("T_total", ""))

        #directory picker
        self.add_output_dir_picker()

        self.output_dir_display.setText(config.get("output_dir", ""))

        #run button
        self.run_button = QPushButton("Comenzar Adquisicion")
        self.layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.on_run_clicked)

        # widgets
        self.setLayout(self.layout)

    # def add_input(self, label_text):
    #     row = QHBoxLayout()
    #     label = QLabel(label_text)
    #     input_field = QLineEdit()
    #     row.addWidget(label)
    #     row.addWidget(input_field)
    #     self.layout.addLayout(row)
    #     return input_field

    def add_input(self, label_text, default_value=""):
        layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(label_text)
        input_field = QtWidgets.QLineEdit()
        input_field.setText(str(default_value))

        # Input fields alignment config
        input_field.setFixedWidth(200)

        layout.addWidget(label)
        layout.addWidget(input_field, alignment=Qt.AlignRight)
        self.layout.addLayout(layout)

        return input_field

    
    ### functions for output directory picking ###
    def add_output_dir_picker(self):
        row = QHBoxLayout()
        label = QLabel("Guardar en")
        self.output_dir_display = QLineEdit()
        self.output_dir_display.setReadOnly(True)

        choose_button = QPushButton("Elegir...")
        choose_button.clicked.connect(self.choose_output_dir)

        row.addWidget(label)
        row.addWidget(self.output_dir_display)
        row.addWidget(choose_button)
        self.layout.addLayout(row)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_display.setText(directory)

    ###
    def on_run_clicked(self):
        print("\n📥 Configuración actual:")

        # inputs
        fs = int(self.fs_input.text())
        chunk_duration = float(self.chunk_duration_input.text())
        threshold = float(self.threshold_input.text())
        T_total = float(self.T_total_input.text())
        output_dir = self.output_dir_display.text()

        # Channels
        channels_text = self.channels_input.text()
        channels = [ch.strip() for ch in channels_text.split(',') if ch.strip()]
        print("channels:", channels)

        # Spectrogram channel name to index
        spectro_channel_name = self.spectro_channel_idx_input.text().strip()
        if spectro_channel_name not in channels:
            print(f"❌ Error: canal '{spectro_channel_name}' no está en {channels}")
            return
        spectro_channel_idx = channels.index(spectro_channel_name)

        # Print parsed values
        print("fs =", fs)
        print("chunk_duration =", chunk_duration)
        print("threshold =", threshold)
        print("T_total =", T_total)
        print("output_dir =", output_dir)
        print("spectro_channel_idx =", spectro_channel_idx, "(from channel:", spectro_channel_name, ")")
        config_data = {
            "fs": fs,
            "chunk_duration": chunk_duration,
            "threshold": threshold,
            "channels": channels_text,
            "spectro_channel": spectro_channel_name,
            "T_total": T_total,
            "output_dir": output_dir
        }

        with open(CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=4) # saves most recent config for next time

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("⚠️ Config file is corrupted, ignoring.")
        return {}



if __name__ == "__main__": # starts app, creates window, shows it, runs event loop
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