# %% nohup python -u train.py > logs/train_$(date +"%Y%m%d_%H%M%S").log 2>&1 & echo $! > train_pid.log
import argparse
from data.pdb.pdb import PDBBindDataset
from src.layers import MoireLayer, get_moire_focus, LigandProjection
from utils.exp import Aliquot, set_device, set_verbose
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import torch
import torch.nn.functional as F

# %%
CONFIG = {
    "MODEL": "Moire",
    "DATASET": "PDBBind",
    "DEPTH": 1,  # args.depth,  # [3 5 8 13 21]
    "MLP_DIM": 384,  # args.dim,
    "HEADS": 16,  # args.heads,
    "FOCUS": "gaussian",
    "DROPOUT": 0.2,
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 5e-4,  # [5e-4, 5e-5] 범위
    "WEIGHT_DECAY": 5e-4,  # lr 줄어드는 속도. 1e-2가 기본
    "T_MAX": 400,  # wandb에서 보고 1 epoch에 들어 있는 step size의 2~3배를 해주세요
    "ETA_MIN": 1e-7,  # lr 최솟값. 보통 조정할 필요 없음.
    "DEVICE": "cuda",
    "SCALE_MIN": 1.0,  # shift 최솟값.
    "SCALE_MAX": 3.5,  # shift 최댓값.
    "WIDTH_BASE": 1.3,  # 보통 조정할 필요 없음.
    "VERBOSE": True,
}

set_verbose(CONFIG["VERBOSE"])
set_device(CONFIG["DEVICE"])
dataset = None
match CONFIG["DATASET"]:
    case "PDBBind":
        dataset = PDBBindDataset()
        criterion = nn.CrossEntropyLoss()
dataset.float()
dataset.batch_size = CONFIG["BATCH_SIZE"]


# %%
class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        dims = config["MLP_DIM"]
        self.ligand_projection = LigandProjection()
        self.input = nn.Sequential(
            nn.Linear(dataset.node_feat_size, dims),
            nn.Linear(dims, dims),
        )
        self.layers = nn.ModuleList(
            [
                MoireLayer(
                    input_dim=dims,
                    output_dim=dims,
                    num_heads=config["HEADS"],
                    shift_min=config["SCALE_MIN"],
                    shift_max=config["SCALE_MAX"],
                    dropout=config["DROPOUT"],
                    focus=get_moire_focus(config["FOCUS"]),
                )
                for _ in range(config["DEPTH"])
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(dims, 192),
            nn.Linear(192, 48),
        )

    def forward(self, x, adj, mask, ligand):
        x = self.input(x)
        for layer in self.layers:
            x = layer(x, adj, mask)
        batch_size, num_nodes, node_feature_size = x.shape
        x = x.view(batch_size * num_nodes, node_feature_size)
        x = self.output(x)
        x = x.view(batch_size, num_nodes, -1)
        ligand_p = self.ligand_projection(ligand)
        ligand_p = ligand_p.unsqueeze(1).repeat(1, x.shape[1], 1)
        interaction = (x * ligand_p).sum(dim=-1)
        interaction = F.relu(interaction)
        return interaction


model = MyModel(CONFIG)
if CONFIG["DEVICE"] == "cuda":
    model = nn.DataParallel(model)
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["T_MAX"], eta_min=CONFIG["ETA_MIN"]
)
Aliquot(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
)(wandb_project="deepInteract", wandb_config=CONFIG, num_epochs=1000, patience=20)