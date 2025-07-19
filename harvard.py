import librosa
import matplotlib.pyplot as plt
import librosa.display

# ✔️ Corrected path to your file
file_path = r"C:\Users\Dr Bia\Desktop\harvard\audiofile.wav"

# ✅ Load the audio file
audio, sr = librosa.load(file_path)

# ✅ Display waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title("Waveform of Audio")
plt.tight_layout()
plt.show()

# ✅ Display MFCC features
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC Features")
plt.tight_layout()
plt.show()

