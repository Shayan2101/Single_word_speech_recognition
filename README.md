# Single_word_speech_recognition
A machine learning project for single-word speech recognition with limited vocabulary.

In this project, we give one-second wav files as input to an MLP neural network.
1- Preproccess.py: This script processes audio files for use in machine learning tasks. It includes functions to load audio data, convert it into Mel-frequency cepstral coefficients (MFCC) features, save these features, and prepare the dataset for training and testing. Additionally, it provides a function to visualize the audio spectrogram.

2- speech_2_convolutions: Audio Classification Using Convolutional Neural Networks (CNN) This project aims to classify audio files into different categories using a Convolutional Neural Network (CNN). The process involves preprocessing the audio data, augmenting it with background noise, and training a CNN model.

3- speech_LSTM.py: This code preprocesses audio data and builds different neural network models (Dense, Conv2D, and LSTM) to classify the audio into predefined categories (e.g., "bed", "bird", "cat", "dog", "down"). It uses TensorFlow and Keras for model building and training and employs TensorBoard for monitoring training progress. The script saves preprocessed audio data into arrays, reshapes the data for model compatibility, trains the models, and evaluates their performance on a test dataset.
