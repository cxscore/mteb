import json
import mteb
import datasets
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check if MPS is available
device = torch.device('mps' if torch.has_mps else 'cpu')

# Custom collate function
def custom_collate_fn(batch):
    sentences1 = [example.texts[0] for example in batch]
    sentences2 = [example.texts[1] for example in batch]
    labels = [example.label for example in batch]
    
    return {'sentence1': sentences1, 'sentence2': sentences2, 'score': torch.tensor(labels, dtype=torch.float)}

# Load training data
train_examples = []
with open('localization.json', 'r') as f:
    for line in f:
        entry = json.loads(line)
        train_examples.append(
            InputExample(texts=[entry['sentence1'], entry['sentence2']], label=float(entry['score']))
        )

# Define the sentence-transformers model
model_name = "avsolatorio/GIST-large-Embedding-v0"
model = SentenceTransformer(model_name).to(device)

# Prepare DataLoader and CosineSimilarityLoss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, collate_fn=custom_collate_fn)
loss_func = losses.CosineSimilarityLoss(model)

# Adam optimizer with learning rate
learning_rate = 2e-5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop parameters
num_epochs = 100
mse_values = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_mse = 0.0
    num_batches = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        # Move inputs to device
        batch_sentences1 = batch['sentence1']
        batch_sentences2 = batch['sentence2']
        batch_labels = batch['score'].to(device)
        
        # Encode sentences
        encoded_texts1 = model.encode(batch_sentences1, convert_to_tensor=True).to(device)
        encoded_texts2 = model.encode(batch_sentences2, convert_to_tensor=True).to(device)
        
        # Compute loss
        loss = loss_func(encoded_texts1, encoded_texts2, batch_labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Compute cosine similarity (which the model tries to predict)
        cosine_sim = F.cosine_similarity(encoded_texts1, encoded_texts2)
        
        # Calculate Mean Squared Error (MSE) for the current batch
        mse = F.mse_loss(cosine_sim, batch_labels)
        epoch_mse += mse.item()
        num_batches += 1
    
    avg_epoch_mse = epoch_mse / num_batches
    mse_values.append(avg_epoch_mse)
    print(f"Epoch {epoch + 1}/{num_epochs}, MSE: {avg_epoch_mse:.4f}")

# Save the fine-tuned model
model.save('fine_tuned_sbert')

# Plotting the MSE over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), mse_values, label='Training MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Training MSE Over Epochs')
plt.legend()
plt.show()

# Optionally re-load the trained model
model = SentenceTransformer('fine_tuned_sbert')

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
