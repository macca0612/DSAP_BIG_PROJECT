# Music Genre Classification using NN methods
DSAP course project, Barcelona School of Telecommunication Engineering (ETSETB), UPC
Authors: Anatolii Skovitin, Francesco Maccantelli
Year: 2023/2024

This repository contains all the code created for this project, which is mainly divided into two sections.
# Process Dataset
Are the files used in order to create the MEL-spectrograms from the audio files:
[process_dataset_img.py](https://github.com/macca0612/DSAP_BIG_PROJECT/blob/main/process_dataset_img.py "process_dataset_img.py")  This script is used to create images for the CNN approach.
[preocess_dataset_data.py](https://github.com/macca0612/DSAP_BIG_PROJECT/blob/main/preocess_dataset_data.py "preocess_dataset_data.py")  This script creates the data used for the hybrid approach.

# Main
[main_alexnet_googlenet.py](https://github.com/macca0612/DSAP_BIG_PROJECT/blob/main/main_alexnet_googlenet.py "main_alexnet_googlenet.py")  This script is used to train the CNN approach. It allows the user to choose between two models for training: AlexNet and GoogLeNet.
[main_cnn_rnn.ipynb](https://github.com/macca0612/DSAP_BIG_PROJECT/blob/main/main_cnn_rnn.ipynb "main_cnn_rnn.ipynb")  The CNN-RNN approach is trained using this script.


