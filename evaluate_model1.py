from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from classifier.data_loader import load_dataset, label2id, id2label
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch
from datasets import DatasetDict

MODEL_ID = "google/flan-t5-small"
REPO_DIR = "flan-t5-small-cybersecurity-text-classification"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset(model_type="AutoModelForSequenceClassification")

# ✅ Step 1: Clean and tokenize text properly
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


# ✅ Step 2: Apply preprocessing to cybersecurity dataset
tokenized_dataset = dataset.map(preprocess_batch, batched=True, remove_columns=dataset["test"].column_names)

# ✅ Step 3: Load model
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    REPO_DIR,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)
model.to(device)
model.eval()

# ✅ Step 4: Create DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
eval_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=1,
    collate_fn=data_collator
)

# ✅ Step 5: Evaluate
preds, labels = [], []
for batch in tqdm(eval_loader):
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy()
        label = batch["labels"].cpu().numpy()
    preds.extend(pred)
    labels.extend(label)

# ✅ Step 6: Compute metrics
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average="weighted")

print("✅ Evaluation on test set:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
