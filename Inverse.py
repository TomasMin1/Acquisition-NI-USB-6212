import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'Tweetie\25-06-2025\Tweetie_25-06-2025_12.35.47.csv')
avg_p = data.iloc[0, 0]   # Column index 2
ampl_p = data.iloc[0, 1]  # Column index 3

from scipy.io import wavfile

samplerate, wav = wavfile.read(r'Tweetie\25-06-2025\sound_Tweetie_25-06-2025_12.35.47.wav')

recovered_signal = (wav*ampl_p)+avg_p

print(ampl_p,avg_p)

t = np.linspace(0, len(recovered_signal) / samplerate, len(recovered_signal))

plt.figure()
plt.plot(t,recovered_signal)
plt.show()