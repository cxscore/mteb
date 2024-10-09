import json
import mteb
import datasets
from sentence_transformers import ( SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer, InputExample, losses) 
from sentence_transformers.losses import CoSENTLoss
from torch.utils.data import DataLoader
import torch

# Check if MPS is available
device = torch.device('mps' if torch.has_mps else 'cpu')

def custom_collate_fn(batch):
    sentences1 = [example.texts[0] for example in batch]
    sentences2 = [example.texts[1] for example in batch]
    labels = [example.label for example in batch]
    
    return {'sentence1': sentences1, 'sentence2': sentences2, 'score': torch.tensor(labels, dtype=torch.float)}

train_examples = []
with open('localization.json', 'r') as f:
    for line in f:
        entry = json.loads(line)
        train_examples.append(
            InputExample(texts=[entry['sentence1'], entry['sentence2']], label=float(entry['score']))
        )

# Define the sentence-transformers model name
model_name = "avsolatorio/GIST-large-Embedding-v0"

# Load the model and move it to the MPS device (if available)
model = SentenceTransformer(model_name).to(device)

# Prepare DataLoader and loss function
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, collate_fn=custom_collate_fn)
loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=100
)
# Save the fine-tuned model
model.save('fine_tuned_sbert')

# Re-load the trained model (optional if continuing from the same script)
model = SentenceTransformer('fine_tuned_sbert')

# Run MTEB evaluation (make sure mteb is defined/imported)
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
