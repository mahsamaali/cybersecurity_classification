import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from classifier.data_loader import load_dataset, label2id, id2label
from torch.utils.data import DataLoader

# === CONFIG ===
MODEL_ID = "google/flan-t5-small"
REPO_DIR = "flan-t5-small-cybersecurity-text-classification1"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load tokenizer and dataset ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset(model_type="AutoModelForSequenceClassification")

# === Preprocessing ===
def preprocess_batch(examples):
    texts = []
    for item in examples["text"]:
        if isinstance(item, list):
            texts.append(" ".join(map(str, item)))
        elif item is None:
            texts.append("")
        else:
            texts.append(str(item))
    encoding = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encoding["labels"] = examples["label"]
    return encoding

tokenized_dataset = dataset.map(preprocess_batch, batched=True, remove_columns=dataset["test"].column_names)

# === Load model ===
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    REPO_DIR,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)
model.to(device)
model.eval()

# === Dataloader ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
eval_loader = DataLoader(tokenized_dataset["test"], batch_size=1, collate_fn=data_collator)

# === Inference ===
preds, labels = [], []
for batch in tqdm(eval_loader, desc="Evaluating"):
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        label = batch["labels"].cpu().numpy()
    preds.extend(pred)
    labels.extend(label)

# === Metrics ===
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")

print("✅ Evaluation on test set:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# === Confusion Matrix ===
used_ids = sorted(set(labels + preds))
label_names = [id2label[i] for i in used_ids]

# Raw confusion matrix
cm = confusion_matrix(labels, preds, labels=used_ids)
df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
df_cm.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))
print("\n📊 Confusion matrix saved to results/confusion_matrix.csv")

# Plot raw confusion matrix
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap="Blues", xticks_rotation=90, values_format='d')
plt.title("Confusion Matrix (Raw Counts)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_raw.png"))
plt.close()

# Normalized confusion matrix
cm_norm = confusion_matrix(labels, preds, labels=used_ids, normalize="true")
plt.figure(figsize=(12, 10))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=label_names)
disp_norm.plot(cmap="Blues", xticks_rotation=90, values_format=".2f")
plt.title("Confusion Matrix (Normalized)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png"))
plt.close()

# === Per-Class Metrics ===
precision, recall, f1_per_class, support = precision_recall_fscore_support(
    labels, preds, labels=used_ids, zero_division=0
)

df_report = pd.DataFrame({
    "label": label_names,
    "precision": precision,
    "recall": recall,
    "f1-score": f1_per_class,
    "support": support
})

df_report.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"), index=False)
print("📄 Classification report saved to results/classification_report.csv")

# === Optional: print the table
print("\n🔍 Per-class classification report:")
print(df_report.round(3))
