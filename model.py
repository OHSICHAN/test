# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from layers import FFN, DeepInteractLayer

# %%
class ProteinProjection(nn.Module):
    def __init__(self):
        super(ProteinProjection, self).__init__()
        self.soft1 = DeepInteractLayer(1536, 768, 128, 7, 7)
        self.soft2 = DeepInteractLayer(768, 384, 64, 5, 5)
        self.soft3 = DeepInteractLayer(384, 192, 32, 4, 4)
        self.soft4 = DeepInteractLayer(192, 96, 32, 3, 3)
        self.soft5 = DeepInteractLayer(96, 48, 16, 3, 3)

    def forward(self, x, adj):
        x = self.soft1(x, adj)
        x = self.soft2(x, adj)
        x = self.soft3(x, adj)
        x = self.soft4(x, adj)
        x = self.soft5(x, adj)
        return x

class LigandProjection(nn.Module):
    def __init__(self):
        super(LigandProjection, self).__init__()
        self.ffn1 = FFN(768, 384, 192)
        self.ffn2 = FFN(192, 192, 192)
        self.ffn3 = FFN(192, 96, 48)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.ffn3(x)
        return x

class DeepInteract(nn.Module):
    def __init__(self):
        super(DeepInteract, self).__init__()
        self.LigandProjection = LigandProjection()
        self.protein_projection = ProteinProjection()

    def forward(self, protein, adj, ligand):
        protein = self.protein_projection(protein, adj)
        ligand = self.LigandProjection(ligand)
        ligand = ligand.unsqueeze(1).repeat(1, protein.shape[1], 1)
        interaction = (protein * ligand).sum(dim=-1)
        interaction = F.elu(interaction)
        return interaction
