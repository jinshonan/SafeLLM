# Extract soft labels from the model
# Batch 4

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, DatasetDict
import numpy as np
from tqdm import tqdm
import time
from datetime import timedelta

# Start timer
start_time = time.time()

# -------------------
# Load Teacher Model
# -------------------
device = "cuda"  # use GPU only

teacher_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
# set a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------
# Load Dataset
# -------------------
train_dataset = load_from_disk("aegis/train")
val_dataset   = load_from_disk("aegis/validation")
test_dataset  = load_from_disk("aegis/test")

# -------------------
# Helper Functions
# -------------------

def make_prompt(user_request: str) -> str:
    return (
        "You are a helpful and concise AI content safety classifier.\n"
        "Given the user's request below, determine if it is harmful.\n"
        "Always answer in this format: 'Yes' or 'No', followed by a brief explanation.\n\n"
        f"User request:\n\"{user_request}\"\n\n"
        "Question: Is this user request harmful?\nAnswer:"
    )

# get probability distribution over "Yes" and "No"
def get_soft_labels_batch(batch):
    prompts = [make_prompt(p) for p in batch["prompt"]]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = teacher_model(**inputs)

    logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No", add_special_tokens=False)[0]

    yes_no_logits = logits[:, [yes_id, no_id]]
    probs = torch.softmax(yes_no_logits, dim=-1)

    return {"soft_labels": probs.cpu().numpy().tolist()}

# -------------------
# Apply to Dataset
# -------------------

def add_soft_labels(dataset):
    return dataset.map(
        get_soft_labels_batch,
        batched=True,
        batch_size=4,   # safe for 3090, can try 8 if memory is fine
        desc="Adding soft labels"
    )

train_dataset = add_soft_labels(train_dataset)
val_dataset   = add_soft_labels(val_dataset)
test_dataset  = add_soft_labels(test_dataset)

# -------------------
# Save Processed Data
# -------------------
processed = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

processed.save_to_disk("aegis_with_softlabels")

# End timer
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600

print("✅ Saved datasets with soft labels at: aegis_with_softlabels")
print(f"⏱️ Total runtime: {elapsed_hours:.2f} hours ({str(timedelta(seconds=int(elapsed_seconds)))})")
