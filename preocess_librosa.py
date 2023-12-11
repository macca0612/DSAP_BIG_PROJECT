import os
import numpy as np
import librosa
from tqdm import tqdm  # Import tqdm for the progress bar

# Path to the folder containing GTZAN dataset audio files
dataset_path = "data/gtzan/genres_original"
output_path = "data/gtzan/data_full"  # Replace with the desired path

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Function to extract and save the MEL spectrogram in multiple temporal segments
def extract_and_save_mel_segments_with_split(file_path, output_folder, segment_split=10, n_mels=128, n_fft=2048, hop_length=512):
    # Load the audio file using Librosa
    y, sr = librosa.load(file_path, sr=None, duration=None)
    
    # Compute the MEL spectrogram for the segment
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Calcola la nuova larghezza per ogni parte
    new_width_per_part = mel_spectrogram.shape[1] // segment_split

    # Inizializza una lista per contenere gli intervalli delle colonne per ciascuna parte
    column_ranges = []
    
    relative_input_path = os.path.relpath(file_path, dataset_path)
    relative_output_path = os.path.join(output_folder, os.path.dirname(relative_input_path))

    # Calcola gli intervalli delle colonne per ciascuna parte
    for part_index in range(segment_split):
        # Calcola la nuova larghezza per questa parte
        part_width = new_width_per_part
        
        # Calcola l'indice di inizio e fine per questa parte
        start_column = part_index * new_width_per_part
        end_column = start_column + part_width
        
        # Aggiungi l'intervallo delle colonne alla lista
        column_ranges.append((start_column, end_column))

    # Ora, puoi utilizzare gli intervalli delle colonne per suddividere lo spettrogramma MEL
    mel_spectrogram_parts = []
    for part_index, (start_column, end_column) in enumerate(column_ranges):
        # Seleziona solo le colonne desiderate per questa parte
        cutted_spectrogram_part = mel_spectrogram[:, start_column:end_column]

        # Resize per garantire che tutte le parti abbiano la stessa lunghezza
        cutted_spectrogram_part_resized = np.resize(cutted_spectrogram_part, (n_mels, new_width_per_part))
        

        # Build the output file path for this part
        part_output_file_path = os.path.join(
            relative_output_path,
            f"{os.path.basename(file_path)}_mel_spectrogram_output.npy"
        )

        # Create subdirectories if they don't exist
        os.makedirs(os.path.dirname(part_output_file_path), exist_ok=True)

        # Save the MEL spectrogram output as a binary file
        np.save(part_output_file_path, cutted_spectrogram_part_resized)


        
def extract_and_save_mel_segments(file_path, output_folder, segment_duration=3, overlap=0.5, n_mels=128, n_fft=2048, hop_length=512):
    # Load the audio file using Librosa
    y, sr = librosa.load(file_path, sr=None, duration=None)

    # Compute the total duration of the audio file
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    relative_input_path = os.path.relpath(file_path, dataset_path)
    
    # reduce the width to specif value
    new_width = 1290
    start_column = (mel_spectrogram.shape[1] - new_width) // 2
    end_column = start_column + new_width

    # Seleziona solo le colonne desiderate
    cutted_spectrogram = mel_spectrogram[:, start_column:end_column]

    # Build the output file path
    relative_output_path = os.path.join(output_folder, os.path.dirname(relative_input_path))
    
    output_file_path = os.path.join(relative_output_path, f"{os.path.basename(file_path)}_mel_spectrogram.npy")

    # Create subdirectories if they don't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the MEL spectrogram segment as a binary file
    np.save(output_file_path, cutted_spectrogram)



if __name__ == "__main__":
    # Process all audio files in the GTZAN dataset
    for root, dirs, files in tqdm(os.walk(dataset_path), desc="Processing files"):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                extract_and_save_mel_segments(file_path, output_path)