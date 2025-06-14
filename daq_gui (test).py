from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
import sys

class DAQConfigWindow(QWidget):
    def __init__(self):
        super().__init__() # initializes Qtwidget
        self.setWindowTitle("Configuracion para la adquisicion de datos")
        self.layout = QVBoxLayout()

        #inputs
        self.fs_input = self.add_input("Frecuencia de sampleo [fs]")
        self.chunk_duration_input = self.add_input("Duracion de los archivos guardados [s]")
        self.threshold_input = self.add_input("Threshold [V]")
        self.channels_input = self.add_input("Canales (separados por coma, ejemplo: ai0,ai1)")
        self.spectro_channel_idx_input = self.add_input("Canal sobre el que se quiere ver su espectrograma (ejemplo: ai0))")
        self.T_total_input = self.add_input("Tiempo de adquisicion total")

        #directory picker
        self.add_output_dir_picker()

        #run button
        self.run_button = QPushButton("Comenzar Adquisicion")
        self.layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.on_run_clicked)

        # widgets
        self.setLayout(self.layout)

    def add_input(self, label_text):
        row = QHBoxLayout()
        label = QLabel(label_text)
        input_field = QLineEdit()
        row.addWidget(label)
        row.addWidget(input_field)
        self.layout.addLayout(row)
        return input_field
    
    ### functions for output directory picking ###
    def add_output_dir_picker(self):
        row = QHBoxLayout()
        label = QLabel("Output Directory")
        self.output_dir_display = QLineEdit()
        self.output_dir_display.setReadOnly(True)

        choose_button = QPushButton("Choose...")
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


if __name__ == "__main__": # starts app, creates window, shows it, runs event loop
    app = QApplication(sys.argv)
    window = DAQConfigWindow()
    window.show()
    sys.exit(app.exec_())

