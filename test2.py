import json
import mteb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

train_examples = []
with open('localization.json', 'r') as f:
    for line in f:
        entry = json.loads(line)
        train_examples.append(
            InputExample(texts=[entry['sentence1'], entry['sentence2']], label=float(entry['score']))
        )

model_name = "avsolatorio/GIST-large-Embedding-v0"

# Load pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Create DataLoader and define the loss function
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Train the model
model.fit(train_objectives=[(train_dataloader)], epochs=100)
model.save('fine_tuned_sbert')
model = SentenceTransformer('fine_tuned_sbert')

s1_list = [example.texts[0] for example in train_examples]
s2_list = [example.texts[1] for example in train_examples]
actual_scores = [example.label for example in train_examples]

# Normalize the actual scores to 0-1 range
min_score, max_score = 0, 5
normalized_scores = [(score - min_score) / (max_score - min_score) for score in actual_scores]

embeddings_s1 = model.encode(s1_list, convert_to_tensor=True)
embeddings_s2 = model.encode(s2_list, convert_to_tensor=True)

# Calculate cosine similarities and store them
cosine_similarities = []
for emb1, emb2 in zip(embeddings_s1, embeddings_s2):
    cos_sim = cosine_similarity(emb1.unsqueeze(0).cpu().numpy(), emb2.unsqueeze(0).cpu().numpy())[0][0]
    cosine_similarities.append(float(cos_sim))

# Calculate Pearson correlation manually between cosine similarities and normalized actual scores
pearson_corr, _ = pearsonr(cosine_similarities, normalized_scores)
print(f"Pearson correlation between cosine similarities and actual scores: {pearson_corr}")

threshold = 0.25
for s1, s2, cos_sim, norm_score in zip(s1_list, s2_list, cosine_similarities, normalized_scores):
    diff = abs(cos_sim - norm_score)
    diff_percentage = diff * 100
    
    if diff > threshold:
        print(f"\nDifference exceeds 25%:")
        print(f"Sentence 1: {s1}")
        print(f"Sentence 2: {s2}")
        print(f"Cosine similarity: {cos_sim:.4f}")
        print(f"Normalized score: {norm_score:.4f}")
        print(f"Difference percentage: {diff_percentage:.2f}%")

# Compute the average difference between cosine similarities and normalized scores
differences = [abs(cos_sim - norm_score) for cos_sim, norm_score in zip(cosine_similarities, normalized_scores)]
avg_difference = np.mean(differences)
avg_difference_percentage = avg_difference * 100
print(f"Average difference percentage: {avg_difference_percentage:.2f}%")

# Check if the average difference is within 25%

if avg_difference <= threshold:
    print(f"Average difference is within the acceptable range of 25%.")
else:
    print(f"Warning: Average difference exceeds the acceptable range of 25%.")

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")