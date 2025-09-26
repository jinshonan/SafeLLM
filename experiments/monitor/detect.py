"""
Simulate a continuous attacker using adversarial artifacts
Self labelling
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import argparse

def classify(checkpoint_path, data_path, device="cuda", max_len=256):
    # Load checkpoint and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    # Load samples - auto-detect file type
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    
    # Debug: Check the structure of your data
    print(f"CSV shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values in second column: {df.iloc[:, 1].isna().sum()}")
    
    # Handle prompts
    prompts = df.iloc[:, 0].astype(str).tolist()
    
    # Handle labels with error checking
    label_column = df.iloc[:, 1]
    print(f"Unique values in label column: {label_column.unique()}")
    
    # Drop rows with missing labels
    if label_column.isna().any():
        print(f"Warning: Found {label_column.isna().sum()} rows with missing labels. Removing them.")
        valid_indices = ~label_column.isna()
        df = df[valid_indices].reset_index(drop=True)
        prompts = df.iloc[:, 0].astype(str).tolist()
        label_column = df.iloc[:, 1]
    
    # Try to convert to integers
    try:
        labels = label_column.astype(int).tolist()
    except (ValueError, pandas.errors.IntCastingNaNError) as e:
        print(f"Error converting labels to int: {e}")
        print("Attempting to convert non-numeric labels...")
        # If labels are text, try to map them to integers
        unique_labels = label_column.unique()
        print(f"Found unique labels: {unique_labels}")
        
        # Create a simple mapping (you may need to adjust this based on your actual labels)
        if len(unique_labels) == 2:
            label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            print(f"Using label mapping: {label_mapping}")
            labels = label_column.map(label_mapping).tolist()
        else:
            raise ValueError(f"Expected 2 unique labels for binary classification, got {len(unique_labels)}: {unique_labels}")
    
    print(f"Final dataset size: {len(prompts)} samples")

    # Tokenize
    encodings = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # probability of "safe" (label=1)
        preds = (probs >= 0.5).int().cpu().numpy()   # default threshold 0.5

    # Metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    # auc = roc_auc_score(labels, probs.cpu().numpy())
    cm = confusion_matrix(labels, preds)

    print("\n📊 Classification Results")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    # print(f"AUC:       {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # Save results to CSV/Excel (matching input format)
    # df["prob_safe"] = probs.cpu().numpy()
    # df["prediction"] = preds
    # if data_path.endswith('.csv'):
    #     out_path = "classification_results.csv"
    #     df.to_csv(out_path, index=False)
    # else:
    #     out_path = "classification_results.xlsx"
    #     df.to_excel(out_path, index=False)
    # print(f"\n✅ Results saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_alpha_0.3")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV or Excel file")
    args = parser.parse_args()

    classify(args.checkpoint, args.data)