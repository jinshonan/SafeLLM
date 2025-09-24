# -----------------------------------------------
# safety distillation with soft labels, notes below
# -----------------------------------------------
# 1. clean_aegis is the part of of the original datasets that match to my test results


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from sklearn.base import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import wandb
import os


# -------------------
# For reproducibility
# -------------------
torch.manual_seed(66)
np.random.seed(66)

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
# Initialize wandb ONCE at the beginning
# -------------------
wandb.init(
    entity="your-entity-name",  # your wandb username/team
    project="deberta-soft-labels",  # replace with your project name
    config={
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "max_length": MAX_LEN,
        "dataset": "clean_aegis"
    }
)


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
    labels = torch.tensor([x["prompt_label"] for x in batch])            # hard labels
    soft_labels = torch.tensor([x["soft_labels"] for x in batch]) # soft labels [p_yes, p_no]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "soft_labels": soft_labels}

train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------
# Model / Optimizer Setup
# -------------------
def init_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
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
    auc = roc_auc_score(all_labels, all_probs)
    
    print(classification_report(all_labels, all_preds))
    print(f"AUC: {auc:.4f}")
    
    return np.mean(losses), acc, auc

# -------------------
# Training Function with α weighting
# -------------------
def train(alpha):
    model, optimizer, scheduler = init_model()
    print(f"\n🚀 Training with α={alpha:.1f} (CE weight), (1-α)={1-alpha:.1f} (KL weight)")

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

        # Save the final model
        save_dir = f"model_alpha_{alpha:.1f}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        val_loss, val_acc, val_auc = evaluate(model, val_loader)

        # Log epoch metrics
        wandb.log({
            f"epoch_train_loss_alpha_{alpha:.1f}": np.mean(epoch_losses),
            f"val_loss_alpha_{alpha:.1f}": val_loss,
            f"val_acc_alpha_{alpha:.1f}": val_acc,
            f"val_auc_alpha_{alpha:.1f}": val_auc,
            "alpha": alpha,
            "epoch": epoch + 1
        })

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Auc: {val_auc:.4f}")

    return model

# -------------------
# Run Experiments for α from 1.0 → 0.0
# -------------------
for alpha in np.linspace(1.0, 0.0, 11):  # 1.0, 0.9, 0.8, ..., 0.0
    train(alpha)


# -------------------
# Finish the wandb run
# -------------------
wandb.finish()