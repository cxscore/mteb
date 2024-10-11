import json
import mteb
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Helper function to find the description in products_desc.json
def find_description(product_name, products_desc):
    for entry in products_desc:
        if entry['Product_name'] == product_name:
            return entry['Description']
    return None

# Load product descriptions from products_desc.json
products_desc = []
with open('products_desc.json', 'r') as f:
    for line in f:
        products_desc.append(json.loads(line))  # Parse each line as a JSON object

# Load sentence pairs and scores from products.jsonl and match descriptions
train_examples = []
with open('products.json', 'r') as f:
    for line in f:
        entry = json.loads(line)  # Parse each line as a JSON object
        sentence1_desc = find_description(entry['sentence1'], products_desc)
        sentence2_desc = find_description(entry['sentence2'], products_desc)
        
        # Only add pairs where both descriptions are found
        if sentence1_desc and sentence2_desc:
            train_examples.append(
                InputExample(texts=[sentence1_desc, sentence2_desc], label=float(entry['score']))
            )

with open('combined_products.json', 'w') as f_out:
    for example in train_examples:
        # Convert InputExample to dictionary format for JSON serialization
        json_line = {
            "sentence1": example.texts[0],
            "sentence2": example.texts[1],
            "score": example.label
        }
        # Write each example as a line in the JSONL file
        f_out.write(json.dumps(json_line) + '\n')

# Define the sentence-transformers model name
model_name = "avsolatorio/GIST-large-Embedding-v0"

# Load pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
