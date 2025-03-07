## Tensor-Flow

## Test Code

The Test Code will Download ope access files from google that will contain audio files of people saying different key words and then we will use those words to create a language model for speech recognition.
The audio files are all shortened to 1 second in length and compiled into 2 different arrays so that we can build the model with the test array and check against it with an analysis array.

# Running Test Code
1. Open command prompt navigate to folder with files readme.md, etc
2. Activate virtual environment if required for python 3.8 - 3.12 for tensor flow requirements with command "start your_env/Scripts/activate"
3. Ensure all required dependencies are installed in your virtual environment see information folder
4. Run Testcode to see language model with python TestCode/main.py




# First look

- Tensor flow is a way to create basic machine learning models
- Tensor is a mathmatical representation of physical properties

- Command to get our audio data into a tensorflow will be keras.utils.audio_dataset_from_directory

# Possible required dependencies

- OS - simple way to use operating simple functionality such as getting data from folders pathlib may also be needed for this (https://docs.python.org/3/library/os.html)
- Seaborn - data visualization library may be needed for personal analysis (https://seaborn.pydata.org/)


# Rough Implementation 

Import of data

DATASET_PATH = 'PATH'

data_dir = pathlib.Path(DATASET_PATH)

//After this you will either need to break data down or change imp to see whole which will likely hurt processing speed

/* Example implementation of a data file (this has output to a test and train dataset so they can compare later) (https://www.tensorflow.org/tutorials/audio/simple_audio)
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')
    */
// Clean up audio files at this point and if using 2 copied sets to compare save 1 and move on with the other

//Cleaning will depend upon raw data information such as number of channels in audio data and other unnecesary information

//After cleaning data can be viewed with Seaborn or other software example uses matplot (matplotlib.pyplot as plt is import) (https://www.tensorflow.org/tutorials/audio/simple_audio)

//Convert waveforms to spectrograms (This may or may not be needed more research required) converts the data from a time domain to a time-frequency-domain
//According to the tutorial for simple audio these files are then fed into the neural network (conversion implementation in https://www.tensorflow.org/tutorials/audio/simple_audio)

## first steps
create an amplitude and frequenc analysis of something like metronomes just so we can tell if a machine is idle or running.
