import json
import mteb
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load training examples from the JSON file
train_examples = []
with open('localization.json', 'r') as f:
    for line in f:
        # Parse each line as a separate JSON object
        entry = json.loads(line)
        train_examples.append(
            InputExample(texts=[entry['sentence1'], entry['sentence2']], label=float(entry['score']))
        )

# Define the sentence-transformers model name
model_name = "avsolatorio/GIST-large-Embedding-v0"

# Load pre-trained sentence transformer model
model = SentenceTransformer(model_name)


# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")

