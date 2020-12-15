# Native-Language-Classification-using-CNN

This repository contains the python code to predict whether the speaker is a native English speaker or not. The dataset used can be found at : [Dataset](https://www.kaggle.com/rtatman/speech-accent-archive)

To run the script in console : python model.py speakers_all.csv

### Preprocessing
The preprocessing.py script performs preprocessing on the dataset. The datset consists of large number of recordings of speakers of different languages. The given model only works on english and arabian recordings. The model can be extended to more languages by modifying the script and including recordings of other languages in the filtered dataset. 

### Model Evaluation
The model_evaluate.py script consists of funtions to predict labels for input, creating confusion matrix on the basis of predictions and finding accuracy.

### Model
The model.py script consists of model definition. It first performs preprocessing on dataset. The filtered dataset consists of .mp3 files. These files are converted to .wav files and stored in audio directory. The .wav files in audio directory are resampled and their MFCCs are obtained. The MFCCs are segmented, train and test dataset is created and the model is trained on the segmented train dataset MFCCs. The model is then tested on test datset and accuracy is determined.

### Performance
100 recordings of english language and 70 recordings of arabian language are included in filtered dataset. The train dataset consists of 83 English speaker recordings and 53 Arabian speaker recordings. The test dataset consists of 17 recordings of each English and Arabian speakers. The overall accuracy of model is found to be 81.25%.
