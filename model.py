import pandas as pd
from collections import Counter
import sys
import os
from pydub import AudioSegment
import preprocessing
from keras import utils
import model_evaluate
import multiprocessing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 50 #20,35

def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index

    y = list(map(lambda x: lang_dict[x],y))
    return utils.to_categorical(y, len(lang_dict))

def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''
    y, sr = librosa.load('../audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array): MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    '''
    Creates segments from mfcc image.
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


def train_model(X_train,y_train,X_validation,y_validation, batch_size=128): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)


    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # Fit model using ImageDataGenerator
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS,
                        callbacks=[es,tb], validation_data=(X_validation,y_validation))

    return (model)

if __name__ == '__main__':
    '''
        Console command example:
        python trainmodel.py bio_metadata.csv model50
        '''

    # Load arguments
    file_name = sys.argv[1]

    # Load metadata
    df = pd.read_csv(file_name)

    # Filter metadata to retrieve only files desired
    filtered_df = preprocessing.data_filter(df)

    print("Filtered Data:")
    print("   ")
    print(filtered_df)

    # Train test split
    X_train, X_test, y_train, y_test = preprocessing.df_split(filtered_df)

    # Get statistics
    train_count = Counter(y_train)
    test_count =  Counter(y_test)

    # To categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Read .mp3 audio files from recordings directory and convert them to .wav files.
    # The .wav files are stored in audio directory.
    destination_folder = 'audio\\'
    source = 'recordings\\'
    for file_name in filtered_df['filename']:
        if not os.path.exists(destination_folder +'{}.wav'.format(file_name)):
            file = file_name
            sound = AudioSegment.from_mp3(source + file + '.mp3')
            sound.export('E:\\python\\Major1\\Final Project\\' + destination_folder + "{}.wav".format(file_name), format="wav")

    # Get resampled wav files using multiprocessing
    if DEBUG:
        print('loading wav files')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)

    # Convert to MFCC
    if DEBUG:
        print('converting to mfcc')
    X_train = pool.map(to_mfcc, X_train)
    X_test = pool.map(to_mfcc, X_test)

    # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train)
    X_validation, y_validation = make_segments(X_test, y_test)

    # Randomize training segments
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0)

    # Train model
    model = train_model(np.array(X_train), np.array(y_train), np.array(X_validation),np.array(y_validation))

    # Make predictions on full X_test MFCCs
    y_predicted = model_evaluate.predict_class_all(create_segmented_mfccs(X_test), model)

    # Print statistics
    print (train_count)
    print (test_count)
    print (np.sum(model_evaluate.confusion_matrix(y_predicted, y_test),axis=1))
    print (model_evaluate.confusion_matrix(y_predicted, y_test))
    print (model_evaluate.get_accuracy(y_predicted,y_test))
