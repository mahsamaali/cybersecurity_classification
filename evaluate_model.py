from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from classifier.data_loader import load_dataset, label2id, id2label

MODEL_ID = "google/flan-t5-small"
REPO_DIR = "flan-t5-small-cybersecurity-text-classification"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
dataset = load_dataset(model_type="AutoModelForSequenceClassification")

# Tokenize the dataset
def preprocess(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoding["label"] = example["label"]
    return encoding

tokenized_dataset = dataset.map(preprocess)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(
    REPO_DIR,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)

# Define safe training arguments
training_args = TrainingArguments(
    output_dir=REPO_DIR,
    per_device_eval_batch_size=1,  # For MPS memory safety
    do_eval=True,
    report_to="none"
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    eval_dataset=tokenized_dataset["test"]
)

# Evaluate
metrics = trainer.evaluate()
print("✅ Evaluation metrics:")
print(metrics)
