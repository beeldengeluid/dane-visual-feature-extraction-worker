
import wave
import numpy as np

from pydub import AudioSegment
from spectrogram.get_spectogram import get_spectrogram

# Convert MP4 to WAV
audio = AudioSegment.from_file("data/train/class_1/test_video.mp4").resample(48000)
audio.export("output.wav", format="wav")

# Read WAV file
wav_file = wave.open("output.wav", 'r')
n_channels, sampwidth, framerate, n_frames = wav_file.getparams()[:4]
data = wav_file.readframes(n_frames)
raw_audio = np.frombuffer(data, dtype=np.int16)

# Segment audio into 1 second chunks
n_samples = raw_audio.shape[1]
n_samples_per_second = 48000
n_samples_per_chunk = n_samples_per_second
n_chunks = int(n_samples / n_samples_per_chunk)
chunks = []
for i in range(n_chunks):
    chunks.append(raw_audio[:, i*n_samples_per_chunk:(i+1)*n_samples_per_chunk])

# Compute spectrogram for each chunk
spectrograms = []
for chunk in chunks:
    spectrograms.append(get_spectrogram(chunk))

