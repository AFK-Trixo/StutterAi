from datasets import Dataset
import pandas as pd
import numpy as np
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch
from torch import nn

# Model and feature extractor names
model_name = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

# Initialize the model for multi-label classification
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=7)

# Update model's loss function for multi-label classification
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
        row['audio'] = audio.squeeze(0).numpy()
        row['sample_rate'] = sample_rate
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        row['audio'] = None
        row['sample_rate'] = None
    return row

df = df.apply(load_example, axis=1)
df = df.dropna(subset=['audio'])

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

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

dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, batched=True)

# Set optimized training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,  # Slightly higher for faster convergence
    per_device_train_batch_size=16,  # Larger batch size for efficiency
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Reduced epochs, combined with higher learning rate
    weight_decay=0.01,
    save_steps=500,  # Adjusted to save periodically during training
    logging_dir='./logs',
    logging_steps=100,  # Reduced logging frequency for efficiency
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
    eval_dataset=dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the model and feature extractor after training
trainer.save_model("./final_model")
feature_extractor.save_pretrained("./final_model")

# Save the training arguments manually as JSON
import json
with open("./final_model/training_args.json", "w") as f:
    json.dump(training_args.to_dict(), f)
