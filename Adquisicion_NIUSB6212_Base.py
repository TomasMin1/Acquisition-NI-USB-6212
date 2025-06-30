from nidaqmx.constants import AcquisitionType, READ_ALL_AVAILABLE
from nidaqmx.stream_readers import AnalogMultiChannelReader
import nidaqmx
import matplotlib.pyplot as plt
import numpy as np

fs = 44150 # Frecuencia de sampleo
chunk_duration = 5 # Duración de cada wav
chunk_samples = int(chunk_duration * fs)

channels = ["ai0",'ai1'] # Canales que se están midiendo
channel_names = ["sound",'pressure'] # Nombre que aparece en el archivo wav correspondiente (Respetar orden de channels)
device = "Dev1" # NO MODIFICAR

with nidaqmx.Task() as task:
    for channel in channels:
        task.ai_channels.add_ai_voltage_chan(device+"/"+ channel)
    
    task.timing.cfg_samp_clk_timing(rate = fs, sample_mode=AcquisitionType.FINITE, samps_per_chan= chunk_samples) # 44150samples/sec×0.01sec=441.5≈441samples per channel
    data = task.read(READ_ALL_AVAILABLE)
    
data = np.array(data)
print(data)
plt.figure()
plt.plot(data[0])
plt.show()

plt.figure()
plt.plot[data[1]]
plt.show()

