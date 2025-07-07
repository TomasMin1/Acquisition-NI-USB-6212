# Acquisition-NI-USB-6212
Code for efficient data acquisition using a NI USB-6212 (DAQ) developed by Sebastián Andribet and Tomás Mininni for use inside the Laboratorio de Sistemas Dinámicos (UBA). The acquisition codes use threading and nidaqmx to communicate with the device. Depending on the code there are 3 (without playback) or 4 (with playback) threads being run simultaneously: an *acquisition_thread* that only acquires from the channels assigned in config and adds the data that satisfies a given condition to an array, a *plotting_thread* that checks if new data has been added to the array,and plots voltage and a spectrogramme as a function of time (for a given channel), a *save_thread* that saves the data as ".wav", and a *playback_thread* that reproduces audio in random order from a folder.

The repository includes a finished code titled *"measurement_and_playback"* that if configured correctly should run endlessly (can be stopped with ctrl+C, more info below), as well as some assorted files (early versions, still usable for testing or finite time measurement).

## measurement_and_playback
Includes three files, adquisicion_global, playback_global y Complete_measurement. To run the code download all three files, and run Complete_measurement. Complete_measurement works as an intermediary for adquisicion_global and playback_global, it allows the user to write a config file (json) that will be used by both codes, and depending on the configurated times, runs either adquisicion_global or playback_global. 

## Asorted Files
- Adquisicion_NIUSB6212_Base.py (single measurement, used for testing)

- Adquisicion y Guardado.py (Loop measurement, does as described above)

- Adquisicion_guardado_playback.py (Loop measurement + audio playback [done via extra thread])

- full_env.yml (used to replicate the enviroment used to make the code via "conda env create -f full_env.yml (full path)")
