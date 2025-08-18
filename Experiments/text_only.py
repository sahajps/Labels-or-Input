import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

model_name = sys.argv[1] # enter model name, e.g., "bert-base-uncased", "roberta-base", etc.
isExt = int(sys.argv[2]) # 0 for original, 1 for extended
RANDOM_SEED = int(sys.argv[3]) # enter random seed for reproducibility
device = sys.argv[4] # enter device, e.g., "cuda:0", "cuda:1", etc.

print("="*30)
print(f"model: {model_name}, isExt: {isExt}, seed: {RANDOM_SEED}, device: {device}")

def seed_setter(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    set_seed(seed)

seed_setter(RANDOM_SEED)

# Load your data
train_df = pd.read_json("../Data/Original/train.jsonl", lines=True)
if(isExt):
    df_train_ex = pd.read_json("../Data/Extended/ex_train.jsonl", lines=True)

    train_df = pd.concat([train_df, df_train_ex], ignore_index=True)

def balance_dataset(df, label_col='label'):
    # Split by class
    class_counts = df[label_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Sample from majority class to match minority count
    df_minority = df[df[label_col] == minority_class]
    df_majority = df[df[label_col] == majority_class].sample(n=len(df_minority), random_state=RANDOM_SEED)

    # Concatenate and shuffle
    balanced_df = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return balanced_df

train_df = balance_dataset(train_df)

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED, stratify=train_df['label'])

test_df_seen = pd.read_json("../Data/Original/test_seen.jsonl", lines=True)
test_df_unseen = pd.read_json("../Data/Original/test_unseen.jsonl", lines=True)
test_df = pd.concat([test_df_seen, test_df_unseen], ignore_index=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Dataset ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

train_dataset = TextDataset(train_df['text'], train_df['label'], tokenizer)
val_dataset = TextDataset(val_df['text'], val_df['label'], tokenizer)
test_dataset = TextDataset(test_df['text'], test_df['label'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- Custom Model ---
class CustomBERTClassifier(nn.Module):
    def __init__(self):
        super(CustomBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        # Freeze all layers except the last encoder layer
        for name, param in self.bert.named_parameters():
            if not any(f'encoder.layer.{i}.' in name for i in [-1, -2]):
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        return logits

model = CustomBERTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# --- Training Loop with Early Stopping ---
best_f1 = 0
patience = 3
counter = 0
num_epochs = 100

# --- Training ---
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(train_loader):.4f}", end=" | ")

    # --- Validation ---
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(true_labels, preds)
    val_f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Val Accuracy: {val_acc:.4f} | weighted-F1: {val_f1:.4f}")

    # --- Early Stopping ---
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        os.makedirs("Chkpts", exist_ok=True)
        torch.save(model.state_dict(), "Chkpts/" + model_name.split("/")[-1] + ".pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# --- Final Evaluation ---
model.load_state_dict(torch.load("Chkpts/" + model_name.split("/")[-1] + ".pt"))
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# --- Printing Results ---
print("\n\n***** Printing Results *****\n\n")
print(f"Classification Report:\n{classification_report(true_labels, preds)}", end="\n\n")

cm = confusion_matrix(true_labels, preds)
print(f"Confusion Matrix:\n{cm}", end="\n\n")

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

print(f"False Positive Rate (FPR): {fpr:.4f}", end="\n")
print(f"False Negative Rate (FNR): {fnr:.4f}", end="\n\n")

print("="*25, end="\n\n\n\n")

# --- Saving Test Predictions ---
os.makedirs("Outputs", exist_ok=True)
res = pd.DataFrame()
res['true_label'] = true_labels
res['predicted_label'] = preds
res.to_csv(f"Outputs/{model_name.split('/')[-1]}_isExt_{isExt}.csv", index=False)