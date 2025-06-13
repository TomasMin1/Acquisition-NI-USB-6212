# Acquisition-NI-USB-6212
Code for efficient data acquisition using a NI USB-6212 (DAQ). Uses threading to run 2 threads: an acquisition_thread that only acquires from the channels assigned in config and adds the data that surpasses a Voltage threshold to an array, and a plotting_thread that checks if new data has been added to the array, plots voltage and a spectrogramme as a function of time and saves the data as .pandas. Uses nidaqmx to communicate with the device.

work in progress
