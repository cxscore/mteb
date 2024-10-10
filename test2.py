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

# Create DataLoader and define the loss function
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
loss = losses.CosineSimilarityLoss(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, loss)], epochs=100)
model.save('fine_tuned_sbert')

# Load the fine-tuned model
model = SentenceTransformer('fine_tuned_sbert')

# Extract the sentences from the training examples
s1_list = [example.texts[0] for example in train_examples]
s2_list = [example.texts[1] for example in train_examples]

# Generate embeddings for each sentence pair
embeddings_s1 = model.encode(s1_list, convert_to_tensor=True)
embeddings_s2 = model.encode(s2_list, convert_to_tensor=True)

# Prepare JSON structure with embeddings and cosine similarities
entries = []
for s1, s2, emb1, emb2 in zip(s1_list, s2_list, embeddings_s1, embeddings_s2):
    # Calculate cosine similarity between each pair of embeddings
    cos_sim = cosine_similarity(emb1.unsqueeze(0).cpu().numpy(), emb2.unsqueeze(0).cpu().numpy())[0][0]
    
    entry = {
        "sentence1": s1,
        "sentence2": s2,
        "embedding1": emb1.cpu().numpy().tolist(),  
        "embedding2": emb2.cpu().numpy().tolist(), 
        "cos_similarity": float(cos_sim) 
    }
    entries.append(entry)

# Save the embeddings and cosine similarities to a JSON file
with open('embeddings_and_similarities.json', 'w') as f:
    json.dump(entries, f, indent=4)

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")

print("Results saved to 'embeddings_and_similarities.json'")
