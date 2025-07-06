import os

import pandas as pd
from datasets import Dataset


#imort from kagle
import kagglehub

#path = kagglehub.dataset_download("ramoliyafenil/text-based-cyber-threat-detection")
#ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = kagglehub.dataset_download("ramoliyafenil/text-based-cyber-threat-detection")
#path + '/cyber-threat-intelligence_all.csv'

# label2id = {"Books": 0, "Clothing & Accessories": 1, "Electronics": 2, "Household": 3}
# id2label = {id: label for label, id in label2id.items()}

labels = [
    'malware', 'attack-pattern', 'TIME', 'identity', 'SOFTWARE',
    'threat-actor', 'location', 'tools', 'FILEPATH', 'SHA2',
    'vulnerability', 'URL', 'DOMAIN', 'IPV4', 'campaign', 'EMAIL',
    'REGISTRYKEY', 'SHA1', 'Infrastucture', 'MD5', 'url', 'hash'
]

# Filter out potential NaN or None values
labels = [label for label in labels if label is not None and str(label).lower() != 'nan']

label2id = {label: idx for idx, label in enumerate(sorted(labels))}
id2label = {idx: label for label, idx in label2id.items()}
# label2id = {'malware':0, 'attack-pattern':1, 'TIME':2, 'identity':3, 'SOFTWARE':4,}
# id2label = {id: label for label, id in label2id.items()}


# label2id = {"Books": 0, "Clothing & Accessories": 1, "Electronics": 2, "Household": 3}
# id2label = {id: label for label, id in label2id.items()}

def load_dataset(model_type: str = "") -> Dataset:
    """Load dataset."""
    dataset_cybersecurity_pandas = pd.read_csv(
        ROOT_DIR + '/cyber-threat-intelligence_all.csv',
        #header=None,
        #names=["label", "text"],
    )
    print(dataset_cybersecurity_pandas.head())
    #clean data 
    #delete none values
    dataset_cybersecurity_pandas=dataset_cybersecurity_pandas.dropna()

    dataset_cybersecurity_pandas["label"] = dataset_cybersecurity_pandas["label"].astype(str)
    if model_type == "AutoModelForSequenceClassification":
        # Convert labels to integers
        dataset_cybersecurity_pandas["label"] = dataset_cybersecurity_pandas["label"].map(
            label2id
        )

    dataset_cybersecurity_pandas["text"] = dataset_cybersecurity_pandas["text"].astype(str)
    dataset = Dataset.from_pandas(dataset_cybersecurity_pandas)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.2)

    return dataset


if __name__ == "__main__":
    print(load_dataset())
