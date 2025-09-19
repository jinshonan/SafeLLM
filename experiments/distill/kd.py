# safety distillation with soft labels

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

# -------------------
# Config
# -------------------
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Load Dataset
# -------------------
datasets = load_from_disk("aegis_with_softlabels")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=MAX_LEN)

tokenized = datasets.map(tokenize_fn, batched=True)

# -------------------
# DataLoaders
# -------------------
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    attention_mask = torch.tensor([x["attention_mask"] for x in batch])
    labels = torch.tensor([x["label"] for x in batch])            # hard labels
    soft_labels = torch.tensor([x["soft_labels"] for x in batch]) # soft labels [p_yes, p_no]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "soft_labels": soft_labels}

train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------
# Model
# -------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Losses
ce_loss = nn.CrossEntropyLoss()  # for hard labels
kl_loss = nn.KLDivLoss(reduction="batchmean")  # for soft labels

# -------------------
# Training Function
# -------------------
def evaluate(model, loader, mode="hard"):
    model.eval()
    correct, total = 0, 0
    losses = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if mode == "hard":
                labels = batch["labels"].to(DEVICE)
                loss = ce_loss(logits, labels)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:  # soft evaluation
                soft_targets = batch["soft_labels"].to(DEVICE)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = kl_loss(log_probs, soft_targets)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == torch.argmax(soft_targets, dim=-1)).sum().item()
                total += soft_targets.size(0)

            losses.append(loss.item())

    acc = correct / total
    avg_loss = np.mean(losses)
    return avg_loss, acc

def train(mode="hard"):
    print(f"\n🚀 Training in {mode}-label mode")
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=False)
        epoch_losses = []

        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            soft_labels = batch["soft_labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if mode == "hard":
                loss = ce_loss(logits, labels)
            else:  # soft labels
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = kl_loss(log_probs, soft_labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            epoch_losses.append(loss.item())
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        val_loss, val_acc = evaluate(model, val_loader, mode=mode)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# -------------------
# Run Experiments
# -------------------
# 1. Train with hard labels
train("hard")

# Re-init model for fair comparison
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# 2. Train with soft labels
train("soft")