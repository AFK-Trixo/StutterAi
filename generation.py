import pandas as pd
import random
import os
from TTS.api import TTS

# Initialize Coqui TTS with a high-quality model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)

# Load the dataset
df = pd.read_csv("synthetic_stuttering_dataset_with_nostutter.csv")

# Create a directory for audio clips if it doesn't exist
output_dir = "audio_clips"
os.makedirs(output_dir, exist_ok=True)

# Define base texts for each stuttering type
base_texts = {
    "Prolongation": "I... I... I need to go now.",
    "Interjection": "Um... well... I think... maybe yes.",
    "Block": "I (pause) want (pause) to (pause) speak.",
    "SoundRep": "So so so many people are here.",
    "WordRep": "I want want want that one.",
    "NoStutteredWords": "This is a clear sentence without any stuttering."
}

# Function to simulate stuttering in the text based on frequency and duration scores
def generate_stuttered_text(base_text, frequency_score, duration_score):
    words = base_text.split()
    stuttered_text = []
    
    # Adjust repetitions based on frequency_score (number of stuttering events)
    for word in words:
        if frequency_score > 1 and random.random() < (frequency_score / 10):
            repetitions = min(frequency_score, 3)  # Cap repetitions to avoid excessive length
            stuttered_text.append(' '.join([word[0] + "-" + word[0] + "-" + word] * repetitions))
        else:
            stuttered_text.append(word)
    
    # Add pauses based on duration_score (length of each stuttered event)
    pause = ""
    if duration_score == 1:
        pause = "..."  # Short pause
    elif duration_score == 2:
        pause = "... ..."  # Moderate pause
    elif duration_score == 3:
        pause = "... ... ..."  # Long pause
    
    # Apply pauses in the stuttered text
    stuttered_text = pause.join(stuttered_text)
    
    return stuttered_text

# Loop through the dataset and generate audio files
for index, row in df.iterrows():
    # Determine stuttering type based on which columns are marked "Present"
    stuttering_type = None
    for stutter_type in ["Prolongation", "Interjection", "Block", "SoundRep", "WordRep"]:
        if row[f"{stutter_type}_Present"] == 1:
            stuttering_type = stutter_type
            break
    # Use a non-stuttered sentence if NoStutteredWords is marked as 1
    if row["NoStutteredWords"] == 1:
        stuttering_type = "NoStutteredWords"

    # Set base text based on stuttering type
    base_text = base_texts.get(stuttering_type, base_texts["NoStutteredWords"])
    
    # Generate stuttered text based on frequency and duration scores, unless it's a non-stuttering clip
    if stuttering_type != "NoStutteredWords":
        frequency_score = row[f"{stuttering_type} Frequency Score"]
        duration_score = row[f"{stuttering_type} Duration Score"]
        stuttered_text = generate_stuttered_text(base_text, frequency_score, duration_score)
    else:
        stuttered_text = base_text

    # Define the filename based on the File ID
    filename = os.path.join(output_dir, f"{row['File ID']}.wav")
    
    # Generate the audio file with Coqui TTS
    print(f"Generating audio for {row['File ID']} with stuttering type: {stuttering_type}")
    tts.tts_to_file(text=stuttered_text, file_path=filename, speed=0.9)  # Adjust speed if necessary

print("Audio generation complete. Files saved in the 'audio_clips' directory.")
