# Acquisition-NI-USB-6212
Code for efficient data acquisition using a NI USB-6212 (DAQ) developed by Sebastián Andribet and Tomás Mininni for use inside the Laboratorio de Sistemas Dinámicos (UBA). Uses threading to run 3 threads: an acquisition_thread that only acquires from the channels assigned in config and adds the data that satisfies a given condition to an array, a plotting_thread that checks if new data has been added to the array,and plots voltage and a spectrogramme as a function of time (for a given channel) and a save_thread that saves the data as ".wav". Uses nidaqmx to communicate with the device.

## Files
Adquisicion_NIUSB6212_Base.py (single measurement, used for testing)
Adquisicion y Guardado.py (Loop measurement, does as described above)
Adquisicion_guardado_playback.py (Loop measurement + audio playback [done via extra thread])
