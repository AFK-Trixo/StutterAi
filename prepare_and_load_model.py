import pandas as pd
import numpy as np
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2CTCTokenizer

# Load your CSV file
csv_file_path = 'C:/Users/faris/OneDrive/Desktop/stuttering-detection/ml-stuttering-events-dataset/SEP-28k_labels_binary.csv'
audio_dir = 'C:/Users/faris/OneDrive/Desktop/stuttering-detection/ml-stuttering-events-dataset/SEP28k_clips'
df = pd.read_csv(csv_file_path)

# Define a function to load each audio file and its corresponding labels
def load_example(row):
    show = row['Show']
    episode = row['EpId']
    clip_id = row['ClipId']
    audio_path = f"{audio_dir}/{show}/{episode}/{show}_{episode}_{clip_id}.wav"
    
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        row['audio'] = audio.squeeze(0).numpy()  # Convert to numpy array
        row['sample_rate'] = sample_rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        row['audio'] = None
        row['sample_rate'] = None
    return row

# Apply the function to load audio data
df = df.apply(load_example, axis=1)

# Remove rows with missing audio files
df = df.dropna(subset=['audio'])

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load processor and model
model_name = "facebook/wav2vec2-large-xlsr-53"

# Load the processor, using explicit tokenizer handling to avoid the symlink issue
processor = Wav2Vec2Processor.from_pretrained(model_name)
try:
    processor.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading tokenizer: {e}. Trying alternative loading method.")
    processor.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name, force_download=True)

# Load the model with the specified number of labels
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=7)  # Adjust num_labels as needed

# Print confirmation
print("Processor and model loaded successfully!")
