from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
from transformers import Trainer, TrainerCallback
from datasets import Dataset

# Assuming `test_dataset` is the Dataset object for the test data
test_dataset = dataset  # Use the same data as evaluation if no separate test set

# Define a compute_metrics function for evaluation
def compute_metrics(pred):
    logits, labels = pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    y_pred = np.where(probs.numpy() > 0.5, 1, 0)  # Threshold at 0.5 for multi-label
    y_true = labels

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Initialize Trainer with evaluation metrics
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Run evaluation
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
