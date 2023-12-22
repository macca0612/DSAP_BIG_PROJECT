# DSAP course project, Barcelona School of Telecommunication Engineering (ETSETB), UPC
# Music Genre Classification using NN methods
# Authors: Anatolii Skovitin, Francesco Maccantelli
# Year: 2023/2024

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set the path to the GTZAN dataset
general_path = 'data/gtzan'

# Function to create and save the full MEL spectrogram
def process_and_save_full_mel(audio_path, output_folder, genre, filename, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Trim silent portions
    y, _ = librosa.effects.trim(y)

    # Calculate MEL spectrogram
    S =  librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=128)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    
    # Create the genre folder if it doesn't exist
    genre_folder = os.path.join(output_folder, genre)
    if not os.path.exists(genre_folder):
        os.makedirs(genre_folder)

    # Plot and save the full MEL spectrogram without title or color bar
    plt.figure(figsize=(16, 6))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='cool')

    # Save the figure without title or color bar
    output_filename = os.path.join(output_folder, f"{genre}/{filename[:-4]}.png")
    plt.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Process each audio file in the dataset
genres = os.listdir(f'{general_path}')
output_folder = 'data/la_primavera/image_mel'

for genre in tqdm(genres):
    genre_path = os.path.join(f'{general_path}', genre)

    for filename in os.listdir(genre_path):
        audio_path = os.path.join(genre_path, filename)
        process_and_save_full_mel(audio_path, output_folder, genre, filename)