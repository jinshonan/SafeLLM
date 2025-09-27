"""
safety distillation with soft labels, notes below

1. for reproduction with a set alpha
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import os


# -------------------
# For reproducibility
# -------------------
torch.manual_seed(88)
np.random.seed(88)

# -------------------
# Config
# -------------------
MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 10
MAX_LEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------
# Load Dataset
# -------------------
datasets = load_from_disk("clean_aegis")

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

    # Map hard labels from str -> int
    label_map = {"unsafe": 0, "safe": 1}
    labels = torch.tensor([label_map[x["prompt_label"]] for x in batch], dtype=torch.long)

    # soft labels [p_unsafe, p_safe]
    soft_labels = torch.tensor([x["soft_labels"] for x in batch], dtype=torch.float32) 
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels, 
        "soft_labels": soft_labels
    }

train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------
# Model / Optimizer Setup
# -------------------
def init_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    print(f"😍 Model loaded successfully on {model.device} with dtype {model.dtype}")

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    return model, optimizer, scheduler

# Loss functions
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")

# -------------------
# Evaluation
# -------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    losses = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            soft_targets = batch["soft_labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Hard loss for reporting
            ce = ce_loss(logits, labels)

            # Accuracy wrt hard labels
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            losses.append(ce.item())

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # probability of positive class

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    
    print(classification_report(all_labels, all_preds))
    print(f"AUC: {auc:.4f}")
    
    return np.mean(losses), acc, auc

# -------------------
# Training Function with α weighting
# -------------------
def train(alpha):
    model, optimizer, scheduler = init_model()
    print(f"\n🚀 Training with α={alpha:.1f} (CE weight), (1-α)={1-alpha:.1f} (KL weight)")
    best_auc = -float("inf")  # for tracking best model for saving

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

            # Hard loss
            ce = ce_loss(logits, labels)

            # Soft loss
            log_probs = torch.log_softmax(logits, dim=-1)
            kl = kl_loss(log_probs, soft_labels)

            # Combined loss
            loss = alpha * ce + (1 - alpha) * kl

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        val_loss, val_acc, val_auc = evaluate(model, val_loader)

        # Save the best model based on validation AUC
        # if val_auc > best_auc:
        #     best_auc = val_auc
        #     model.save_pretrained(f"best_model_alpha_{alpha:.1f}")
        #     tokenizer.save_pretrained(f"best_model_alpha_{alpha:.1f}")

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Auc: {val_auc:.4f}")

    return model

# -------------------
# reproduce α = 3
# -------------------
alpha = 3
train(alpha)