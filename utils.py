# %%
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


# Custom collate function to handle dynamic padding
def collate_batch(batch):
    node_features, adj_matrices, ligands, targets = zip(*batch)
    # Find the maximum number of nodes in the current batch
    max_nodes = max([nf.size(0) for nf in node_features])
    # Pad node features and adjacency matrices
    padded_node_features = [
        F.pad(nf, (0, 0, 0, max_nodes - nf.size(0))) for nf in node_features
    ]
    padded_adj_matrices = [
        F.pad(adj, (0, max_nodes - adj.size(0), 0, max_nodes - adj.size(1)))
        for adj in adj_matrices
    ]
    padded_targets = [F.pad(tgt, (0, max_nodes - tgt.size(0))) for tgt in targets]
    # Stack all padded tensors
    node_features = torch.stack(padded_node_features)
    adj_matrices = torch.stack(padded_adj_matrices)
    targets = torch.stack(padded_targets)
    ligands = torch.stack(ligands)
    return node_features, adj_matrices, ligands, targets


# Sample Dataset class with padding
class GraphDataset(Dataset):
    def __init__(self, node_features, adjacency_matrices, ligands, targets):
        """
        Args:
            node_features: List of node feature tensors, one tensor per graph.
            adjacency_matrices: List of adjacency matrices, one matrix per graph.
            targets: List of target values, one per graph.
            max_graph_size: The maximum number of nodes any graph can have.
        """
        self.node_features = node_features
        self.adjacency_matrices = adjacency_matrices
        self.ligands = ligands
        self.targets = targets

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        return (
            self.node_features[idx],
            self.adjacency_matrices[idx],
            self.ligands[idx],
            self.targets[idx],
        )

    def DataLoader(self, batch_size, shuffle=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            pin_memory=True,
        )


def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch_idx, (node_features, adj_matrix, ligand, targets) in enumerate(
            dataloader
        ):
            node_features = node_features.to(device)
            adj_matrix = adj_matrix.to(device)
            ligand = ligand.to(device)
            targets = targets.to(device)
            outputs = model(node_features, adj_matrix, ligand)
            outputs = outputs.to(torch.float32)
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            total_eval_loss += loss.item()
    return total_eval_loss / len(dataloader)


def load_best_model(model, device="cpu"):
    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=torch.device(device=device),
            weights_only=True,
        ),
    )
    return model
