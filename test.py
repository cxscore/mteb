import json
import mteb
from datasets import Dataset, DatasetInfo
from sentence_transformers import SentenceTransformer, InputExample, losses, SentenceTransformerTrainer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformerTrainingArguments
import torch
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState, TrainerControl,get_linear_schedule_with_warmup


# Check if MPS is available
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')

# Load training data
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
    num_train_epochs=100,  # Adjust the number of epochs
    per_device_train_batch_size=16,  # Batch size for training
    learning_rate=5e-6,  # Learning rate
    warmup_ratio=0.1,  # Warmup for the learning rate scheduler
    save_strategy="steps",  # Save model after every epoch
    logging_steps=100,  # Log every 100 steps
    save_total_limit=2,  # Number of maximum checkpoints to save
    run_name="fine_tuned_sbert_run"  # Tracking run name for logging
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

    # Called every time there is a log update
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])  # Log the loss
            if state.is_local_process_zero:
                print(f"Step: {state.global_step}, Loss: {logs['loss']:.4f}")

# Instantiate the logger
loss_logger = LossLoggerCallback()

# Trainer with custom logging callback and arguments
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,  # Use the training arguments
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # Use the same dataset for evaluation
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

