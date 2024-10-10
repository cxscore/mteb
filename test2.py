import json
import mteb
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, SentenceTransformerTrainer
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

train_examples = []
with open('localization.json', 'r') as f:
    for line in f:
        # Parse each line as a separate JSON object
        entry = json.loads(line)
        train_examples.append(
            InputExample(texts=[entry['sentence1'], entry['sentence2']], label=float(entry['score']))
        )

# Define the sentence-transformers model name
#model_name = "average_word_embeddings_komninos"
# or directly from huggingface:
model_name = "avsolatorio/GIST-large-Embedding-v0"


model = SentenceTransformer(model_name)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, loss)], epochs=100)
model.save('fine_tuned_sbert')

# Load the fine-tuned model
model = SentenceTransformer('fine_tuned_sbert')

# Extract the sentences from the train_examples
s1_list = [example.texts[0] for example in train_examples]
s2_list = [example.texts[1] for example in train_examples]

# Generate embeddings for each sentence pair
embeddings_s1 = model.encode(s1_list, convert_to_tensor=True)
embeddings_s2 = model.encode(s2_list, convert_to_tensor=True)

# Calculate cosine similarity between each pair of sentences
cos_similarities = cosine_similarity(embeddings_s1.cpu().numpy(), embeddings_s2.cpu().numpy())

entries = []
for idx, (s1, s2, emb1, emb2, cos_sim) in enumerate(zip(s1_list, s2_list, embeddings_s1, embeddings_s2, cos_similarities)):
    entry = {
        "sentence1": s1,
        "sentence2": s2,
        "embedding1": emb1.cpu().numpy().tolist(),  # Convert tensor to list
        "embedding2": emb2.cpu().numpy().tolist(),  # Convert tensor to list
        "cos_similarity": cos_sim  # Cosine similarity
    }
    entries.append(entry)

print(type(entries))
# Save to JSON file
with open('embeddings_and_similarities.json', 'w') as f:
    json.dump(entries, f, indent=4)

model = SentenceTransformer('fine_tuned_sbert')

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")