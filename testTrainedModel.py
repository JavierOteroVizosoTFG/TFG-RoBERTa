import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, matthews_corrcoef
import random
import numpy as np

# Set the seed for reproducibility
seed = 26
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Define paths to the model and tokenizer
model_path = "/path/to/your/model/directory"
tokenizer_path = "/path/to/your/tokenizer/directory"

# Load the pre-trained model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

# Define the path to your new data
new_data_path = "/path/to/your/new_data.csv"

# Read the new data from the CSV file
new_df = pd.read_csv(new_data_path, delimiter=';')

batch_size = 16

# Tokenize the new data
new_encodings = tokenizer(
    new_df["Titulo"].tolist(),
    new_df["Descripcion"].tolist(),
    new_df["Fecha"].tolist(),
    padding="max_length",
    truncation='only_second',
    max_length=128,
    return_tensors="pt"
)

new_input_ids = new_encodings["input_ids"]
new_attention_masks = new_encodings["attention_mask"]

new_dataset = TensorDataset(new_input_ids, new_attention_masks)
new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()

predictions = []
with torch.no_grad():
    for batch in new_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks = batch

        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits

        _, predicted_labels = torch.max(logits, 1)
        predictions.extend(predicted_labels.tolist())

true_labels = new_df["Label"].tolist()

accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
mcc = matthews_corrcoef(true_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"MCC: {mcc:.4f}")
