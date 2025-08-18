from transformers import AutoModel, AutoProcessor, set_seed
import pandas as pd
from PIL import Image
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import sys

model_name = sys.argv[1] # enter model name, e.g., "google/vit-base-patch16-224", etc.
isExt = int(sys.argv[2]) # 0 for original, 1 for extended
RANDOM_SEED = int(sys.argv[3]) # enter random seed for reproducibility
device = sys.argv[4] # enter device, e.g., "cpu", "cuda:0", "cuda:1", etc.

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

processor = AutoProcessor.from_pretrained(model_name)

# --- Dataset ---
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image_path = os.path.join(self.img_dir, 'Original/'+row['img'])
            image = Image.open(image_path).convert("RGB")
        except:
            image_path = os.path.join(self.img_dir, 'Extended/'+row['img'])
            image = Image.open(image_path).convert("RGB")
        text = row['text']
        label = torch.tensor(row['label'], dtype=torch.long)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": label
        }

train_dataset = ImageDataset(train_df, "../Data", processor)
val_dataset = ImageDataset(val_df, "../Data", processor)
test_dataset = ImageDataset(test_df, "../Data", processor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# --- Custom Model ---
class CustomViTClassifier(nn.Module):
    def __init__(self):
        super(CustomViTClassifier, self).__init__()
        self.vit = AutoModel.from_pretrained(model_name)

        for param in self.vit.parameters():
            param.requires_grad = False

        for name, param in self.vit.encoder.layer[-2:].named_parameters():
            param.requires_grad = True

        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.vit(
            pixel_values=pixel_values
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_embedding)
        return logits

model = CustomViTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# --- Training Loop with Early Stopping ---
best_f1 = 0
patience = 3
counter = 0
num_epochs = 100

# Training
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}", end=" | ")
    model.train()
    total_loss = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        logits = model(pixel_values)
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
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            logits = model(pixel_values)
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(true_labels, preds)
    val_f1 = f1_score(true_labels, preds, average='weighted')
    print(f"Val Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")

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
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        logits = model(pixel_values)
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