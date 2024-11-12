import os 
import librosa, librosa.display
import matplotlib.pyplot as plt 
import numpy as np

file = "C:\\Users\\GOOD\Desktop\\Komil\\audio_preprocessing\\11-preprocessing_audio_data_for_deep_learning\\blues.wav"


# waveform
signal, sr = librosa.load(file, sr=22050)  # sr*T --> 22050 * 30
librosa.display.waveshow(signal, sr = sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# fft --> spectogram
fft = np.fft.fft(signal)
magnitude =  np.abs(fft)
frequncy = np.linspace(0, sr, len(magnitude))
plt.plot(frequncy, magnitude)
plt.xlabel("Frequncy")
plt.ylabel("magnitude")
plt.show()

# stft --> spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrgram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrgram)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
 
# MFCCs
MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr = sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()