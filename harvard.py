import librosa
import matplotlib.pyplot as plt
import librosa.display

# Corrected path to your file
file_path = r"C:\Users\Dr Bia\Desktop\harvard\audiofile.wav"

# Load the audio file
audio, sr = librosa.load(file_path)

#  Display waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title("Waveform of Audio")
plt.tight_layout()
plt.show()

#  Display MFCC features
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC Features")
plt.tight_layout()
plt.show()
# Step 2: Tamper the same audio (simulate splicing)
# Use raw string for Windows path
sound = AudioSegment.from_wav(r"C:\Users\Dr Bia\Desktop\harvard\audiofile.wav")

# Cut first 2 seconds and paste at the end
spliced = sound[2000:] + sound[:2000]

# Export tampered version
spliced.export(r"C:\Users\Dr Bia\Desktop\harvard\tampered_audiofile.wav", format="wav")

# Step 3: Load and display features of tampered audio
tampered_audio, sr2 = librosa.load(r"C:\Users\Dr Bia\Desktop\harvard\tampered_audiofile.wav")

# Display waveform of tampered
plt.figure(figsize=(10, 4))
librosa.display.waveshow(tampered_audio, sr=sr2)
plt.title("Waveform of Tampered Audio")
plt.tight_layout()
plt.show()

# Display MFCC of tampered
tampered_mfcc = librosa.feature.mfcc(y=tampered_audio, sr=sr2, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(tampered_mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC of Tampered Audio")
plt.tight_layout()
plt.show()
