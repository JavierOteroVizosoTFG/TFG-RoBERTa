import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd

# Load the saved model and tokenizer
model_path = "/path/to/your/model/directory"
tokenizer_path = "/path/to/your/tokenizer/directory"

model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

# Input news article
titulo = input("Please enter the headline of the news: ")
descripcion = input("Please enter the news description: ")
fecha = input("Please enter the news date (DD/MM/YYYY): ")

# Preprocess the input news article using the tokenizer
input_encodings = tokenizer(
    titulo,
    descripcion,
    fecha,
    padding="max_length",
    truncation='only_second',
    max_length=128,
    return_tensors="pt"
)

input_ids = input_encodings["input_ids"]
attention_masks = input_encodings["attention_mask"]

# Perform inference with the loaded model on the input news article
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    outputs = model(input_ids.to(device), attention_mask=attention_masks.to(device))
    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    probability_of_being_true = probabilities[0, 1].item()*100  # Get probability of the "true" class

# Print the calculated veracity of the input news article
print(f"News Veracity: {probability_of_being_true:.2f} %")
