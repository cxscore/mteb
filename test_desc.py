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
with open('products_desc.json', 'r') as f:
    products_desc = json.load(f)

# Load sentence pairs and scores from products.json and match descriptions
train_examples = []
with open('products.json', 'r') as f:
    products = json.load(f)
    for entry in products:
        sentence1_desc = find_description(entry['sentence1'], products_desc)
        sentence2_desc = find_description(entry['sentence2'], products_desc)
        
        # Only add pairs where both descriptions are found
        if sentence1_desc and sentence2_desc:
            train_examples.append(
                InputExample(texts=[sentence1_desc, sentence2_desc], label=float(entry['score']))
            )

# Define the sentence-transformers model name
model_name = "avsolatorio/GIST-large-Embedding-v0"

# Load pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Create DataLoader and define the loss function
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
loss = losses.CosineSimilarityLoss(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, loss)], epochs=100)
model.save('fine_tuned_sbert')

# Load the fine-tuned model
model = SentenceTransformer('fine_tuned_sbert')

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")

print("Results saved to 'embeddings_and_similarities_desc.json'")
