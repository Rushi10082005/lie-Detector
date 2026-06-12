# models/text_model.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This part runs when the module is first imported.
# It downloads and loads the model, which can take time.
print("Loading Text Model (BERT)... This may take a moment.")
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
print("Text Model loaded successfully.")

def text_model_predict(texts):
    """
    Gets predictions from the loaded BERT model.
    Args:
        texts (list of str): A list of text statements to analyze.
    Returns:
        numpy.ndarray: An array of probabilities for [truth, lie] for each text.
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    
    # Run model in inference mode
    with torch.no_grad():
        outputs = text_model(**inputs)
        
    logits = outputs.logits
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)
    
    return probabilities.cpu().numpy()