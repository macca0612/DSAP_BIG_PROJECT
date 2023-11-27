import os
import numpy as np
import librosa
from tqdm import tqdm  # Import tqdm for the progress bar

# Path to the folder containing GTZAN dataset audio files
dataset_path = "data/gtzan/genres_original"
output_path = "data/gtzan/data_MEL_3_0.5"  # Replace with the desired path

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Function to extract and save the MEL spectrogram in multiple temporal segments
def extract_and_save_mel_segments(file_path, output_folder, segment_duration=3, overlap=0.5, n_mels=128, n_fft=2048, hop_length=512):
    # Load the audio file using Librosa
    y, sr = librosa.load(file_path, sr=None, duration=None)

    # Compute the total duration of the audio file
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the number of segments
    num_segments = int(total_duration / segment_duration)

    # Calculate the hop length based on the overlap
    hop_length = int(segment_duration * (1 - overlap) * sr)

    for i in range(num_segments):
        # Extract the audio segment
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        segment = y[int(start_time * sr):int(end_time * sr)]

        # Compute the MEL spectrogram for the segment
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

        # Get the relative path of the input file
        relative_input_path = os.path.relpath(file_path, dataset_path)

        # Build the output file path
        relative_output_path = os.path.join(output_folder, os.path.dirname(relative_input_path))
        output_file_path = os.path.join(relative_output_path, f"{os.path.basename(file_path)}_segment_{i}_mel_spectrogram.npy")

        # Create subdirectories if they don't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Save the MEL spectrogram segment as a binary file
        np.save(output_file_path, mel_spectrogram)

# Process all audio files in the GTZAN dataset with tqdm progress bar
for root, dirs, files in tqdm(os.walk(dataset_path), desc="Processing files"):
    for file in tqdm(files):
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            extract_and_save_mel_segments(file_path, output_path)