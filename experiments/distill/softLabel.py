# Extract soft labels from the model

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, DatasetDict
import numpy as np
from tqdm import tqdm

# -------------------
# Load Teacher Model
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

teacher_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

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
def get_soft_labels(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = teacher_model(**inputs)

    # get logits for the *next token*
    logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    # extract logits for "Yes" and "No"
    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

    yes_logit = logits[yes_id].item()
    no_logit = logits[no_id].item()

    probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=-1).cpu().numpy()

    return {"soft_labels": probs.tolist()}  # [p_yes, p_no]

# -------------------
# Apply to Dataset
# -------------------

def add_soft_labels(dataset):
    return dataset.map(
        lambda x: get_soft_labels(make_prompt(x["prompt"])),
        batched=False
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

print("✅ Saved datasets with soft labels at: aegis_with_softlabels")
