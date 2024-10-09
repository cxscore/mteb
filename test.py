import json
import mteb
import datasets
from sentence_transformers import SentenceTransformer, InputExample, losses, SentenceTransformerTrainer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
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

# Custom callback to log training loss
class LossLoggerCallback:
    def __init__(self):
        self.losses = []

    def __call__(self, loss_value, epoch, step):
        # Log the loss for each step
        self.losses.append(loss_value)
        if step % 100 == 0:  # Optionally print loss every 100 steps
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value:.4f}")

# Instantiate the logger
loss_logger = LossLoggerCallback()

# Training loop parameters
num_epochs = 100

# Trainer with custom logging callback
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_examples,
    loss=loss_func,
    optimizer=optimizer,
    epochs=num_epochs,
    warmup_steps=100,  # Optional: for learning rate warmup
    callback=loss_logger  # Log the losses
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save('fine_tuned_sbert')

# Plotting the loss after training
plt.figure(figsize=(10, 6))
plt.plot(loss_logger.losses, label='Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.legend()
plt.show()

# Optionally re-load the trained model
model = SentenceTransformer('fine_tuned_sbert')

# Run MTEB evaluation
tasks = mteb.get_tasks(tasks=["CXS-STS"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/{model_name}")
