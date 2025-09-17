#!/usr/bin/env python
# coding: utf-8

# # DistilBERT Binary Classification
# This notebook fine-tunes a DistilBERT model to classify prompts as either `safe` or `unsafe`.
# 
# - Uses Hugging Face Transformers and Datasets
# - Runs on Apple M1 using PyTorch with MPS backend
# - Includes evaluation with F1 score and confusion matrix

from datasets import load_from_disk
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set device to MPS (Apple Silicon GPU) if available
device = torch.device('cuda' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the dataset
train_dataset = load_from_disk("aegis/train")
val_dataset = load_from_disk("aegis/validation")
test_dataset = load_from_disk("aegis/test")

# change it to integers
label_map = {"safe": 0, "unsafe": 1}
train_dataset = train_dataset.map(lambda x: {"labels": label_map[x["prompt_label"]]})  # shit has to be "labels" in plural??
val_dataset = val_dataset.map(lambda x: {"labels": label_map[x["prompt_label"]]})
test_dataset = test_dataset.map(lambda x: {"labels": label_map[x["prompt_label"]]})

# Quick look at dataset format
train_dataset[0]

# debug
# print("Column names:", train_dataset.column_names)
# print("First example:", train_dataset[0])
# print("Labels type:", type(train_dataset[0]["labels"]))
# print("Labels value:", train_dataset[0]["labels"])

# Tokenize the dataset
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Verify it worked
print("After set_format - first example keys:", train_dataset[0].keys())


# tokenize debug:

print("Column names after tokenization:", train_dataset.column_names)
print("Keys in first example:", train_dataset[0].keys())

# Check if labels exist in the dataset
print("Does 'labels' column exist?", 'labels' in train_dataset.column_names)
print("First few labels:", train_dataset['labels'][:5])
print("Type of labels column:", type(train_dataset['labels']))

# Check the actual first example more thoroughly
first_example = train_dataset[0]
print("All keys in first example:", list(first_example.keys()))
if 'labels' in first_example:
    print("Labels value:", first_example['labels'])
else:
    print("Labels key is missing from first example!")


# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)


# Define compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=50,
    push_to_hub=False,
)

# # Minimal training arguments - should work without accelerate issues
# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     push_to_hub=False,
# )


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()


# Evaluate on test set
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Confusion matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "unsafe"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
plt.savefig('eval1.png', dpi=300, bbox_inches='tight')


# Extended evaluation
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay

# Compute ROC-AUC (if model outputs probabilities)
probs = predictions.predictions[:, 1]  # Probability for class 1 (unsafe)
roc_auc = roc_auc_score(labels, probs)
print(f"ROC-AUC: {roc_auc:.4f}")

# ROC Curve
RocCurveDisplay.from_predictions(labels, probs)
plt.title("ROC Curve")
plt.show()
plt.savefig('eval2.png', dpi=300, bbox_inches='tight')

# Detailed classification report
print("Classification Report:")
print(classification_report(labels, preds, target_names=["safe", "unsafe"]))