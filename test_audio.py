import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch

# Load the trained model and feature extractor
model_path = "./final_model"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define the path to your audio file
audio_file_path = r"C:\Users\faris\OneDrive\Desktop\stuttering-detection\ml-stuttering-events-dataset\SEP28k_clips\WomenWhoStutter\1\WomenWhoStutter_1_0.wav"

# Load the audio file
def load_audio(file_path):
    audio, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    return audio.squeeze().numpy(), 16000  # Flatten and return audio data and sample rate

def predict_stuttering(audio_file_path, threshold=0.5):
    # Load and preprocess audio
    audio, sample_rate = load_audio(audio_file_path)
    
    # Extract features
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    
    # Run the model on the input and get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Print raw logits for debugging
    print("Raw Logits:", logits)
    
    # Apply sigmoid to convert logits to probabilities for multi-label classification
    probabilities = torch.sigmoid(logits).squeeze(0).tolist()
    
    # If probabilities is nested, flatten it
    if isinstance(probabilities[0], list):
        probabilities = [item for sublist in probabilities for item in sublist]
    
    # Define the stuttering types based on the labels in your dataset
    stuttering_types = ["Prolongation", "Block", "SoundRep", "WordRep", "NaturalPause", "NoStutteredWords", "Interjection"]
    
    # Classify each stuttering type based on the threshold
    results = {}
    print("Stuttering Type Classification:")
    for stuttering_type, prob in zip(stuttering_types, probabilities):
        status = "Detected" if prob >= threshold else "Not Detected"
        results[stuttering_type] = status
        print(f"{stuttering_type}: {status} (Probability: {prob:.4f})")

    return results
# Run prediction on your audio file
results = predict_stuttering(audio_file_path, threshold=0.3)
