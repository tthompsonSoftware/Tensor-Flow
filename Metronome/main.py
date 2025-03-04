# Beat tracking example
#import keyword
import os
import pathlib
#from tkinter import CHAR

import matplotlib.pyplot as plt
import numpy as np
import librosa
#import seaborn as sns
#import tensorflow as tf
#from scipy import signal

# extensions not sure if needed
#from tensorflow.keras import layers
#from tensorflow.keras import models
#from IPython import display

# NOTES
# I believe that a mfcc will be good once i undesrtand it because from my basic understanding it helps to analyze audio
# in a more human sense. So it will help get data changes that are more relevant. Still not sure.
#
# I need to still grab amplitude and frequency data currently showing bpm and frames(still dont fully understand frames
# they seem to be arbitrary segments of time for analysis not a showing of the actual beat).



# 1. Get the file path to an included audio example C:\Users\tthom\source\repos\Tensor-Flow\Metronome\Audio_examples
#filename = librosa.example('Metronome/Audio_examples/metronome-1.mp3')
#filename = librosa.example('C:/Users/tthom/source/repos/Tensor-Flow/Metronome/Audio_examples/metronome-1.mp3')

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
#y, sr = librosa.load("metronome-1.mp3", duration=30)
y, sr = librosa.load("Metronome/Audio_examples/metronome-1.mp3", duration=60)

#2.5 Seperates percussive noice bashing and clanging from harmonic noise like a violin
y_harmonic, y_percussive = librosa.effects.hpss(y)

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print(beat_times)

print(librosa.feature.mfcc(y=y, sr=sr))

#fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(1,4))
#plt.plot(segment, color='black')
#plt.show()

#librosa.display.waveshow(y)
     
#plt.plot(y, 'audio', 'time', 'amplitude')
#fig, ax = plt.subplots(nrows=3, sharex=True)
#librosa.display.waveshow(y, sr=sr, ax=ax[0])
#ax[0].set(title='Envelope view, mono')
#ax[0].label_outer()


# regular plot for data on waveform
fig, ax = plt.subplots(sharex=True)
librosa.display.waveshow(y, sr=sr, ax=ax)
ax.set(title='Envelope view, mono')
ax.label_outer()
#plt.show()

# Shows harmonic and percussive across the entire diagram may be useful for tensorflow analysis when mahcine is running
# currently not very useful.
fig, ax = plt.subplots(sharex=True)
y_harm, y_perc = librosa.effects.hpss(y)
librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax, label='Harmonic')
librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax, label='Percussive')
ax.set(title='Multiple waveforms')
ax.legend()
#plt.show()

# Good for showing the Harmonic and percussive information during a specific time frame change xlim to get different time
# change ylim to get higher and lower amplitude.
fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
ax.set(xlim=[5.0, 7.0], title='Sample view', ylim=[-0.2, 0.2])
librosa.display.waveshow(y, sr=sr, ax=ax, marker='.', label='Full signal')
librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
ax.label_outer()
ax.legend()
ax2.legend()
#plt.show()

#display amplitude
waveform = np.array(y)

# Find the maximum amplitude
max_amplitude = np.max(waveform)
print(f"Maximum amplitude: {max_amplitude}")

# Find the maximum absolute amplitude
max_abs_amplitude = np.max(np.abs(waveform))
print(f"Maximum absolute amplitude: {max_abs_amplitude}")

waveform = np.array(y)

# Find the minimum amplitude
min_amplitude = np.min(waveform)
print(f"minimum amplitude: {min_amplitude}")

# Find the maximum absolute amplitude
min_abs_amplitude = np.min(np.abs(waveform))
print(f"minimum absolute amplitude: {min_abs_amplitude}")

def plot(vector, name, xlabel=None, ylabel=None):
    plt.figure()
    plt.plot(vector)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()
    plt.savefig('static/plots/' + name)