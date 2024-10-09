import json
import mteb
import datasets
from sentence_transformers import SentenceTransformer, InputExample, losses, SentenceTransformerTrainer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformerTrainingArguments
import torch
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

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

# Set up training arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir="fine_tuned_sbert",
    num_train_epochs=10,  # Adjust the number of epochs
    per_device_train_batch_size=16,  # Batch size for training
    learning_rate=2e-5,  # Learning rate
    warmup_ratio=0.1,  # Warmup for the learning rate scheduler
    save_strategy="epoch",  # Save model after every epoch
    logging_steps=100,  # Log every 100 steps
    evaluation_strategy="steps",  # Evaluation strategy
    eval_steps=100,  # Evaluate every 100 steps
    save_total_limit=2,  # Limit to 2 model saves
    fp16=True,  # Enable 16-bit precision if supported
    run_name="fine_tuned_sbert_run"  # Tracking run name for logging
)

# AdamW optimizer with learning rate
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

# Scheduler: Linear warmup followed by linear decay
num_train_steps = len(train_dataloader) * training_args.num_train_epochs
warmup_steps = int(training_args.warmup_ratio * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

# Custom callback to log training loss
class LossLoggerCallback:
    def __init__(self):
        self.losses = []

    def __call__(self, loss_value, epoch, step):
        self.losses.append(loss_value)
        if step % 100 == 0:  # Optionally print loss every 100 steps
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value:.4f}")

# Instantiate the logger
loss_logger = LossLoggerCallback()

# Trainer with custom logging callback and arguments
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,  # Use the training arguments
    train_dataset=train_examples,
    loss=loss_func,
    optimizers=(optimizer, scheduler),  # Pass optimizer and scheduler
    callbacks=[loss_logger]  # Log the losses
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
