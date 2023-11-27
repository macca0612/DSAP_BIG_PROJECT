import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the GTZAN dataset
general_path = 'data/gtzan'


# Function to create and save the full MEL spectrogram
def process_and_save_full_mel(audio_path, output_folder, genre, filename, n_fft=2048, hop_length=512, n_mels=128):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Trim silent portions
    y, _ = librosa.effects.trim(y)

    # Calculate MEL spectrogram
    # Define n_mels, n_fft, hop_size
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)

    # Create the genre folder if it doesn't exist
    genre_folder = os.path.join(output_folder, genre)
    os.makedirs(genre_folder, exist_ok=True)  # exist_ok=True will prevent any error if the directory already exists

    # Plot and save the full MEL spectrogram without title or color bar
    plt.figure(figsize=(16, 6))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='inferno')

    # Save the figure without title or color bar
    output_filename = os.path.join(output_folder, f"{genre}/{genre}.{filename[:-4]}.png")
    plt.axis('off')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()


# Process each audio file in the dataset
genres = os.listdir(f'{general_path}/genres_original')
output_folder = 'data/gtzan/images_MEL'

# Create the folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for genre in genres:
    genre_path = os.path.join(f'{general_path}/genres_original', genre)
    print(f"Now in the folder: {genre}")
    if not os.path.isdir(genre_path):
        print(f"Skipped non-directory path: {genre_path}")
        continue

    for filename in os.listdir(genre_path):
        if filename.lower().endswith('.wav'):
            audio_path = os.path.join(genre_path, filename)
            process_and_save_full_mel(audio_path, output_folder, genre, filename)
