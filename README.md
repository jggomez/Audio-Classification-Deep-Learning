# Audio Classification for Deaf Individuals

## Notebook
[Notebook](https://colab.research.google.com/drive/1EiUf__69bQ2rj9xtDCigqTq8ajzBcggO?usp=sharing)

## Description
This repository contains the code and resources for an audio classification project aimed at assisting deaf individuals. The primary goal is to develop a robust model that can classify common household sounds such as a 'door knock', 'running water', 'chair dragging', and 'human voice'. By classifying these sounds, the model can provide critical information to a deaf person about potential situations occurring in their environment, thereby enhancing their situational awareness and safety.

## Data Description
The dataset used for training and testing this model consists of audio samples categorized into four distinct classes:
*   **Chair**: Sounds of chairs being dragged or moved.
*   **Door**: Sounds related to doors, such as knocks or opening/closing.
*   **Voice**: Human speech or vocalizations.
*   **Water**: Sounds of running water, like a faucet being turned on.

The audio files are in WAV format and have been preprocessed to ensure consistency and extract relevant features. The dataset was created by the authors using [Edge Impulse](https://www.edgeimpulse.com/) and a mobile phone to record the audios.

<img width="1652" height="807" alt="Screenshot 2026-04-18 at 1 56 05 p m" src="https://github.com/user-attachments/assets/d645cb7d-31d6-4fc1-a02c-0fd408102664" />

<img width="1190" height="770" alt="Screenshot 2026-04-18 at 1 55 41 p m" src="https://github.com/user-attachments/assets/43de8c48-7bca-4bd6-a1c6-f6d847265068" />

## Feature Extraction: Spectrograms and MFCC

### Spectrograms
Initially, audio waves were transformed into **spectrograms** (Short-Time Fourier Transforms). Spectrograms visualize the frequency content of a sound as it changes over time, providing a 2D representation suitable for convolutional neural networks. A `preprocess` function was developed to handle varying audio lengths by padding shorter files with zeros and truncating longer ones to a consistent length of 16000 samples.

### MFCC (Mel-Frequency Cepstral Coefficients)
To capture more perceptually relevant features, the project later transitioned to using **Mel-Frequency Cepstral Coefficients (MFCCs)**. MFCCs compactly represent the spectral envelope of an audio signal, making them highly effective for speech and sound recognition. We extracted 20 MFCCs along with their first (delta) and second (delta-delta) derivatives, resulting in 60 features per time step. This provides a richer representation of how the sound's characteristics change over its duration.

## Models Used

### 1. Convolutional Neural Network (CNN)
An initial CNN model (`modelo_conv`) was implemented using Keras. It consisted of:
*   Two `Conv2D` layers with `BatchNormalization` and `ReLU` activation.
*   `MaxPooling2D` layers to reduce dimensionality.
*   A `GlobalAveragePooling2D` layer.
*   A `Dropout` layer (initially 0.6, reduced to 0.3).
*   A final `Dense` layer with `softmax` activation for classification.

This model was trained on the spectrogram features with `l2` regularization (initially 0.01, reduced to 0.001) and `EarlyStopping` callbacks. The goal was to identify spatial patterns in the spectrograms.

<img width="825" height="310" alt="audio-classi-conv2 drawio" src="https://github.com/user-attachments/assets/2bb06662-5c40-408e-a055-68fef620cfd6" />

### 2. Recurrent Neural Network (RNN) with LSTM
Recognizing the sequential nature of audio data, a second model (`modelo_rnn`) based on an RNN with LSTM layers was developed. This model was trained using MFCCs (including deltas and delta-deltas) and aimed to capture temporal dependencies in the audio signals. The architecture includes:
*   `Input` layer for the 60 MFCC features over time.
*   `BatchNormalization` to stabilize inputs.
*   Two `LSTM` layers (`64` and `32` units) to process sequential data.
*   A `Dropout` layer (0.5).
*   A final `Dense` layer with `softmax` activation.

Optimization techniques like `ReduceLROnPlateau` and `EarlyStopping` were incorporated during training.

<img width="712" height="290" alt="Audio-Classification-LSTM drawio" src="https://github.com/user-attachments/assets/fe2324b6-56d3-4bec-b881-de75185c3b67" />

## Results

### CNN Model Performance
Initial results with the CNN model on spectrograms showed an accuracy of approximately **52%** on the validation set. While the model showed some learning, there was significant confusion, particularly between classes 0, 1, and 2, and class 0 was rarely predicted. This suggested that simple spectrograms combined with CNNs were not optimally capturing the discriminative features for all classes.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/0c58a5b3-0d48-4fd9-8c16-20e6c44aa887" />

<img width="511" height="413" alt="image" src="https://github.com/user-attachments/assets/94a284e1-1fbc-4745-a250-fc0f51517507" />

### RNN LSTM Model Performance (MFCC + Augmentation)
By switching to MFCC features and employing an LSTM-based RNN, the model performance significantly improved. The validation accuracy reached approximately **74%**. Key improvements were observed:
*   **Class Discrimination**: The model showed excellent performance for Class 3 (water), achieving a high F1-score (0.96), indicating that its unique temporal/spectral signatures were well-captured.
*   **Sequential Learning**: The LSTM architecture successfully processed the 1-second audio duration as a cohesive sequence.
*   **Generalization**: Data augmentation (white noise injection) and `BatchNormalization` helped prevent overfitting, leading to more generalized learning.

Despite improvements, some confusion still persists among Classes 0, 1, and 2, suggesting these sounds share similar frequency profiles that require further investigation or more advanced feature engineering.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/f04bc6b5-3032-484a-8516-bfd69213fd2d" />

<img width="511" height="413" alt="image" src="https://github.com/user-attachments/assets/bb2de55e-2925-468d-8815-08aec7ea20fb" />

## References
*   [Keras Documentation](https://keras.io/)
*   [TensorFlow Documentation](https://www.tensorflow.org/)
*   [Librosa Documentation](https://librosa.org/doc/latest/index.html)
*   [Scikit-learn Documentation](https://scikit-learn.org/stable/)
