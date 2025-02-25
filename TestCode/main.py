# Imports of dependencies
import keyword
import os
import pathlib
from tkinter import CHAR

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy import signal


# extensions not sure if needed
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


# Start of implementation

# Path set for file to load
DATASET_PATH = 'data/mini_speech_commands_extracted/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

  print(data_dir)

# take data examples and label by type of data ex. (Running well, Stopped, Startup, etc) types depend on wants by Josh

operation = np.array(tf.io.gfile.listdir(str(data_dir)))
operation = operation[(operation != 'README.md') & (operation != '.DS_Store')] # Data folder will need a README.md with command info in gfile format
print('operation:', operation) # Print to test works

# Split dataset in order to have a test and validation array

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000, # determined based off length of clips 16000 is 1 second sets data to known length for easier comparison might not work
    subset='both')

    

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

# looks like this simply shows the type of tensor shape

train_ds.element_spec

# prepares the data for reading by the machine(?) tf.squeeze reduces dataset to 1 channel

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# seperates the data into an array to seperate (DONT OVERWRITE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# show audio wavefors by data quantity

for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

# show opertation names of the dataset

label_names[[1,1,3,0]]

# Plot audio waveforms on a diagram

plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
  plt.subplot(rows, cols, i+1)
  audio_signal = example_audio[i]
  plt.plot(audio_signal)
  plt.title(label_names[example_labels[i]])
  plt.yticks(np.arange(-1.2, 1.2, 0.2))
  plt.ylim([-1.1, 1.1])

# Test to this point before moving to machine learning
# Function doesnt go here but affects below this
# Creates a spectrogram from waveform data

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

# Pickup code here
# using function show spectrogram data as text and allow for audio playback

for i in range(3):
  label = label_names[example_labels[i]]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=16000))

# Function to show spectrogram as an image

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# use plot spectrogram function to show waveform and spectrogram to check the correlation

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
#plt.show()

# function to create dataset from spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# examine ???

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

# Show all spectrogram plots

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

#plt.show()

# build and train

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# Create model for testing it is a CNN

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

# configure the model

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# Train

EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

# Evaluate curves

metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

model.evaluate(test_spectrogram_ds, return_dict=True)

# Display a confusion matrix

y_pred = model.predict(test_spectrogram_ds)

y_pred = tf.argmax(y_pred, axis=1)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
plt.close('all')

check = 0
x = ""
answer = ""
path = ""
keyword = ""
file = ""
filenames = ""
directory = 'data/mini_speech_commands_extracted/mini_speech_commands/'
n = 1
num = 0

while check != 2:
    answer = input('Provide keyword of file to test against if you are finished type stop.\n')
    if answer != "stop":
        # run this
        keyword = answer
        try:
            filenames = os.listdir(directory + keyword)
            for filename in filenames:
                print(str(n) + '. ' + filename)
                n += 1
            selection = input('select item #\n')
            num = int(selection)

            file = filenames[num]
        except FileNotFoundError:
            print(f"Error: Directory '{directory}' not found.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

        x = 'data/mini_speech_commands_extracted/mini_speech_commands/' + keyword + '/' + file
        x = tf.io.read_file(str(x))
        x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
        x = tf.squeeze(x, axis=-1)
        waveform = x
        x = get_spectrogram(x)
        print(x.shape)
        fig, axes = plt.subplots(2, figsize=(12, 8))
        plot_spectrogram(x, axes[0])
        axes[0].set_title('Spectrogram')
        plt.suptitle(label.title())
        plt.show()
        fs = 240
        f, t, Sxx = signal.spectrogram(x, fs)
        print(f)
        x = x[tf.newaxis,...]

        prediction = model(x)
        x_labels = label_names
        plt.bar(x_labels, tf.nn.softmax(prediction[0]))
        plt.title('No')
        plt.show()
        print('Evaluation of data.\n')
        #print(model.evaluate(x,return_dict=True))
        
    else:
        break