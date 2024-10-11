import json
import mteb
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Define the sentence-transformers model name
model_name = "avsolatorio/GIST-large-Embedding-v0"

# Load pre-trained sentence transformer model
model = SentenceTransformer(model_name)


# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")

