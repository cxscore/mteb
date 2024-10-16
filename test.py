import json
import mteb
from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from torch.optim import AdamW
import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState, TrainerControl, get_linear_schedule_with_warmup
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')

s1_list, s2_list, score_list = [], [], []
with open('localization.json', 'r') as f:
    for line in f:
        entry = json.loads(line)
        s1_list.append(entry['sentence1'])
        s2_list.append(entry['sentence2'])
        score_list.append(entry['score'])

model_name = "avsolatorio/GIST-large-Embedding-v0"
model = SentenceTransformer(model_name).to(device)

train_dataset = Dataset.from_dict({
    "sentence1": s1_list,
    "sentence2": s2_list,
    "label": score_list,
})

loss_func = losses.CosineSimilarityLoss(model)

# Set up training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="fine_tuned_sbert",
    num_train_epochs=100,  
    per_device_train_batch_size=16,  
    learning_rate=5e-6,  
    warmup_ratio=0.1,  
    save_strategy="steps",  
    logging_steps=100,  
    save_total_limit=2,  
    run_name="fine_tuned_sbert_run"  
)

# AdamW optimizer with learning rate
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

# Scheduler: Linear warmup followed by linear decay
num_train_steps = len(train_dataset) * training_args.num_train_epochs
warmup_steps = int(training_args.warmup_ratio * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])  
            if state.is_local_process_zero:
                print(f"Step: {state.global_step}, Loss: {logs['loss']:.4f}")

loss_logger = LossLoggerCallback()

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,  
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  
    loss=loss_func,
    optimizers=(optimizer, scheduler),  
    callbacks=[loss_logger]  
)

trainer.train()

model.save('fine_tuned_sbert')

# Plotting the loss after training
plt.figure(figsize=(10, 6))
plt.plot(loss_logger.losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.legend()
plt.show()

# Reload the trained model
model = SentenceTransformer('fine_tuned_sbert')

embeddings_s1 = model.encode(s1_list, convert_to_tensor=True)
embeddings_s2 = model.encode(s2_list, convert_to_tensor=True)

# Normalize the actual scores to 0-1 range
min_score, max_score = 0, 5
normalized_scores = [(score - min_score) / (max_score - min_score) for score in score_list]

# Calculate cosine similarities and store them
cosine_similarities = []
for emb1, emb2 in zip(embeddings_s1, embeddings_s2):
    cos_sim = cosine_similarity(emb1.unsqueeze(0).cpu().numpy(), emb2.unsqueeze(0).cpu().numpy())[0][0]
    cosine_similarities.append(float(cos_sim))

# Calculate Pearson correlation manually between cosine similarities and normalized actual scores
pearson_corr, _ = pearsonr(cosine_similarities, normalized_scores)
print(f"Pearson correlation between cosine similarities and actual scores: {pearson_corr}")

# Check each cosine similarity score for differences greater than 25%
threshold = 0.25
for s1, s2, cos_sim, norm_score in zip(s1_list, s2_list, cosine_similarities, normalized_scores):
    diff = abs(cos_sim - norm_score)
    diff_percentage = diff * 100
    
    # If the difference exceeds 25%, print details
    if diff > threshold:
        print(f"\nDifference exceeds 25%:")
        print(f"Sentence 1: {s1}")
        print(f"Sentence 2: {s2}")
        print(f"Cosine similarity: {cos_sim:.4f}")
        print(f"Normalized score: {norm_score:.4f}")
        print(f"Difference percentage: {diff_percentage:.2f}%")

# Compute the overall average difference
differences = [abs(cos_sim - norm_score) for cos_sim, norm_score in zip(cosine_similarities, normalized_scores)]
avg_difference = np.mean(differences)
avg_difference_percentage = avg_difference * 100
print(f"\nAverage difference percentage: {avg_difference_percentage:.2f}%")

# Check if the average difference is within 25%
if avg_difference <= threshold:
    print(f"Average difference is within the acceptable range of 25%.")
else:
    print(f"Warning: Average difference exceeds the acceptable range of 25%.")

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
