import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from scipy.io import wavfile

class SEP28kDataset(Dataset):
    def __init__(self, labels_csv, audio_dir, transform=None):
        # Initialize variables
        self.labels_df = pd.read_csv(labels_csv)  # Load the labels file into a DataFrame
        self.audio_dir = audio_dir  # Directory where audio files are stored
        self.transform = transform  # Optional transforms for data augmentation or processing

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get a sample by index (idx)
        row = self.labels_df.iloc[idx]  # Retrieve the row for this index
        
        # Extract metadata for the audio file
        show = row['Show']
        ep_id = row['EpId']
        clip_id = row['ClipId']
        
        # Construct the file path to the audio file
        file_path = os.path.join(self.audio_dir, f"{show}/{ep_id}/{show}_{ep_id}_{clip_id}.wav")
        file_path = os.path.normpath(file_path)  # Normalize the path
        
        # Debugging output for the file path
        print(f"Attempting to load file at: {file_path}")
        
        # Load the audio file
        try:
            sample_rate, waveform = wavfile.read(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None, None  # Return a tuple of None values if the file is not found

        # Convert labels to a tensor
        labels = torch.tensor(row[['Prolongation', 'Block', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords', 'NaturalPause']].astype(float).values, dtype=torch.float)

        # Apply any transforms to the waveform (if specified)
        if self.transform:
            waveform = self.transform(waveform)

        # Return the audio waveform and labels as a tuple
        return torch.tensor(waveform, dtype=torch.float), labels

# Test the dataset class
if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = SEP28kDataset(labels_csv='SEP-28k_labels_binary.csv', audio_dir='C:/Users/faris/OneDrive/Desktop/stuttering-detection/ml-stuttering-events-dataset/SEP28k_clips')

    # Test by loading a sample
    sample = dataset[0]
    
    if sample[0] is not None and sample[1] is not None:
        sample_audio, sample_labels = sample
        print("Sample audio shape:", sample_audio.shape)
        print("Sample labels:", sample_labels)
    else:
        print("Sample not available due to missing audio file.")
