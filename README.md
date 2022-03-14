# Classification of Electroencephalography (EEG) Signals using Neural Networks

Course project for ECE 247. 

Dataset: BCI 4 dataset comprising of 2115 trials corresponding to 22 EEG channels over 1000 time bins

Preprocessing:
- Trimming
- Averaging
- Max pooling
- Noise addition (data augmentation)
- LP Filter for smoothing EEG

Train Test Split:
- Random sampling
- Stratification (across class and labels)

Models:
- CNN
- RNN/ GRU
- LSTM
- CNN + LSTM
- CNN + RNN
- VAE + Logistic Regression
