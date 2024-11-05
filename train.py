# %%
"""
USAGE
nohup python train.py | tee train.log & disown
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import (
    GraphDataset,
    evaluate_model,
)
from model import DeepInteract
import wandb
import random
from settings import PROJECT_NAME, WANDB, BATCH_SIZE, EVALUATION_SIZE, LEARNING_RATE, NUM_EPOCHS, CUTOFF, PATIENCE, LOAD_MODEL, get_device

# SETTINGS
device = get_device(verbose=True)

def save_best_model(model, eval_loss, best_eval_loss):
    if eval_loss < best_eval_loss:
        # Save the model with the wandb name
        torch.save(model.state_dict(), wandb.run.name + ".pth")
        return eval_loss
    return best_eval_loss


# Training Loop
def train_model(
    model,
    train_dataloader,
    eval_dataloader,
    criterion,
    optimizer,
    num_epochs=100,
    patience=PATIENCE,
):
    model.train()
    best_eval_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (node_features, adj_matrix, ligand, targets) in enumerate(
            train_dataloader
        ):
            optimizer.zero_grad()
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)
            ligand = ligand.to(device)
            targets = targets.to(device)
            outputs = model(node_features, adj_matrix, ligand)
            # output dtype to torch.float32
            outputs = outputs.to(torch.float32)
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            loss.backward()
            # NOTE: Gradient clipping
            max_norm = 1.0  # Define the maximum norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            total_loss += loss.item()

            # Log the loss to WandB
            if WANDB:
                wandb.log({"Loss/train": loss.item()})

        eval_loss = evaluate_model(model, eval_dataloader, criterion, device)

        if eval_loss > 1:
            continue

        # Check for improvement
        if eval_loss < best_eval_loss:
            save_best_model(
                model, eval_loss, best_eval_loss
            )  # Update the best model if needed
            best_eval_loss = eval_loss
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1

        # Log evaluation loss
        if WANDB:
            wandb.log({"Loss/eval": eval_loss, "Best Loss/eval": best_eval_loss})

        print(
            f"Epoch {epoch+1:04}/{num_epochs:04} - Loss: {total_loss/len(train_dataloader):.4f} - Eval Loss: {eval_loss:.4f}"
        )

        # Early stopping condition
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Finish the WandB run
    if WANDB:
        wandb.finish()


DIR = "/Users/saankim/Documents/dev/DeepInteract/dataset/{}"
DIR = "/home/bioscience/dev/DeepInteract/features/all/{}"

node_features = torch.load(
    DIR.format("node.pth"),
    map_location=torch.device(device="cpu"),
    # weights_only=True,
)[:CUTOFF]
node_features = [n.squeeze(0) for n in node_features]
adjacency_matrices = torch.load(
    DIR.format("adj.pth"),
    map_location=torch.device(device="cpu"),
    # weights_only=True,
)[:CUTOFF]
# move to device
ligands = torch.load(
    DIR.format("mol.pth"),
    map_location=torch.device(device="cpu"),
    # weights_only=True,
)[:CUTOFF]
targets = torch.load(
    DIR.format("regr.pth"),
    map_location=torch.device(device="cpu"),
    # weights_only=True,
)[:CUTOFF]
# remove datapoints with len(node_features) > 600 or len(node_features) < 100
indices = [i for i, nf in enumerate(node_features) if 100 <= len(nf) <= 600]
node_features = [node_features[i] for i in indices]
adjacency_matrices = [adjacency_matrices[i] for i in indices]
ligands = [ligands[i] for i in indices]
targets = [targets[i] for i in indices]
targets = [t - 1 for t in targets]  # 1 옹스트롬 이내는 강한 interaction
targets = [t / 30 for t in targets]
targets = [torch.log(t) for t in targets]
targets = [t * -t.abs() for t in targets]
targets = [F.elu(t) for t in targets]
targets = [torch.clamp(t, -2, 2) for t in targets]

# display histogram of a target
# import matplotlib.pyplot as plt

# Iterate over targets to create cumulative histogram over all targets
# for target in targets[:100]:
#     plt.hist(target.to("cpu").numpy(), alpha=0.3, bins=20)
#     plt.xlabel("Distance Value")
#     plt.ylabel("Count")

# plt.show()

# Sample number of random datapoints for training and evaluation
indices = list(range(len(node_features)))
random.shuffle(indices)
eval_indices = indices[:EVALUATION_SIZE]
train_indices = indices[EVALUATION_SIZE:]

# Sanity check
print(f"Number of datapoints: {len(node_features)}")
assert (
    len(node_features) == len(adjacency_matrices) == len(ligands) == len(targets)
), "The number of datapoints are not equal"
assert (
    len(train_indices) + len(eval_indices) == len(node_features)
), "The split is not correct"
assert (
    len(train_indices) % BATCH_SIZE == 0
), "The batch size is not a factor of the training set"

# %% Train Run
# Create dataloaders
train_dataloader = GraphDataset(
    [node_features[i] for i in train_indices],
    [adjacency_matrices[i] for i in train_indices],
    [ligands[i] for i in train_indices],
    [targets[i] for i in train_indices],
).DataLoader(batch_size=BATCH_SIZE)
eval_dataloader = GraphDataset(
    [node_features[i] for i in eval_indices],
    [adjacency_matrices[i] for i in eval_indices],
    [ligands[i] for i in eval_indices],
    [targets[i] for i in eval_indices],
).DataLoader(batch_size=BATCH_SIZE)

# Initialize model, loss function, and optimizer
model = DeepInteract()
if LOAD_MODEL:
    model.load_state_dict(torch.load(LOAD_MODEL))
    print(f"Model loaded from {LOAD_MODEL}")
model.to(device)

# Initialize loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize WandB run
if WANDB:
    wandb.init(project=PROJECT_NAME)

# Train the model
train_model(
    model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
)
