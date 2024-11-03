import torchaudio
import numpy as np
from datasets import Dataset
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from torch import nn
import time  # For adding delays if needed
import os

# Model and feature extractor paths
model_name = "./final_model"  # Load from `final_model`
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Update model's configuration for multi-label classification
model.config.problem_type = "multi_label_classification"

# Load and prepare the dataset
csv_file_path = 'C:/Users/faris/OneDrive/Desktop/stuttering-detection/ml-stuttering-events-dataset/SEP-28k_labels_binary.csv'
audio_dir = 'C:/Users/faris/OneDrive/Desktop/stuttering-detection/ml-stuttering-events-dataset/SEP28k_clips'
df = pd.read_csv(csv_file_path)

# Function to load and process each audio file and its corresponding labels
def load_example(row):
    show = row['Show']
    episode = row['EpId']
    clip_id = row['ClipId']
    audio_path = f"{audio_dir}/{show}/{episode}/{show}_{episode}_{clip_id}.wav"
    
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        row['audio'] = audio.squeeze().numpy()
        row['sample_rate'] = 16000
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        row['audio'] = None
        row['sample_rate'] = None
    return row

# Apply audio loading to the DataFrame and drop any rows with errors
df = df.apply(load_example, axis=1)
df = df.dropna(subset=['audio'])

# Convert DataFrame to Hugging Face Dataset and apply random sampling
dataset = Dataset.from_pandas(df.sample(frac=0.3, random_state=42))  # Using 30% of data for quick training

# Data collator for padding and processing
data_collator = DataCollatorWithPadding(feature_extractor)

# Preprocess audio function
def preprocess_function(examples):
    audio = examples["audio"]
    inputs = feature_extractor(audio, sampling_rate=16000, padding=True, truncation=True, max_length=32000)
    labels = np.array([
        examples['Prolongation'], examples['Block'], examples['SoundRep'],
        examples['WordRep'], examples['NaturalPause'], examples['NoStutteredWords'],
        examples['Interjection']
    ]).T
    inputs["labels"] = labels.tolist()
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./final_model",  # Overwrite the model in `final_model`
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,  # Using smaller batch size
    gradient_accumulation_steps=2,  # Accumulate to simulate a batch size of 8
    num_train_epochs=3,  # Limiting to fewer epochs for quicker training
    weight_decay=0.01,
    fp16=True,  # Mixed precision for RTX 4050
    save_steps=2000,  # Reduced save frequency to minimize interruptions
    logging_dir='./logs',
    logging_steps=200,  # Reduced logging frequency to minimize interruptions
    overwrite_output_dir=True  # Allows overwriting the model in the same directory
)

# Custom trainer to override default loss function with BCEWithLogitsLoss
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

# Initialize Trainer with modified loss function
trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Using the same dataset for quick evaluation
    data_collator=data_collator,
)

# Start training
trainer.train()

# Clear GPU cache (if needed)
torch.cuda.empty_cache()
time.sleep(2)  # Ensure everything is closed properly

# Primary save attempt
try:
    model.save_pretrained("./final_model")
    feature_extractor.save_pretrained("./final_model")
    print("Model and feature extractor saved successfully.")
except Exception as e:
    print(f"Primary save failed: {e}")
    
    # Fallback save with a different directory name
    fallback_dir = "./backup_final_model"
    os.makedirs(fallback_dir, exist_ok=True)
    model.save_pretrained(fallback_dir)
    feature_extractor.save_pretrained(fallback_dir)
    print("Model saved to backup directory.")

# Optionally save training arguments and state
trainer.state.save_to_json("./final_model/trainer_state.json")
training_args.to_json("./final_model/training_args.json")
