# %%
from utils import load_best_model
from model import DeepInteract
from IPython.display import display
import torch.nn.functional as F
import torch

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda:0" if torch.cuda.is_available() else "cpu"
)
display(device)

model = DeepInteract()
model.to(device)
model = load_best_model(model, device)
model.eval()

# Load input data
DIR = "/Users/saankim/Documents/dev/DeepInteract/dataset/{}"
node_features = torch.load(
    DIR.format("node_0-100.pth"),
    map_location=torch.device(device=device),
    weights_only=True,
)[:1]
node_features = [n.squeeze(0) for n in node_features]
adjacency_matrices = torch.load(
    DIR.format("adj_0-100.pth"),
    map_location=torch.device(device=device),
    weights_only=True,
)[:1]
ligands = torch.load(
    DIR.format("mol_0-100.pth"),
    map_location=torch.device(device=device),
    weights_only=True,
)[:1]
targets = torch.load(
    DIR.format("regr_0-100.pth"),
    map_location=torch.device(device=device),
    weights_only=True,
)[:1]
targets = [t / 30 for t in targets]
targets = [torch.log(t) for t in targets]
targets = [t * -t.abs() for t in targets]
targets = [F.elu(t) for t in targets]
input_data = (
    node_features[0].unsqueeze(0),
    adjacency_matrices[0].unsqueeze(0),
    ligands[0].unsqueeze(0),
)

# Get prediction result
# input_data = input_data.to(device)
with torch.no_grad():
    output = model(*input_data)
    # prediction = torch.argmax(output, dim=1)

import seaborn as sns
import matplotlib.pyplot as plt

target = targets[0].unsqueeze(0)
comparison = torch.cat([output, target], dim=0)
comparison = comparison.cpu().numpy()

# visualize the output and targets in an heatmap
# NOTE: 아래쪽이 답지
plt.figure(figsize=(10, 0.8))
sns.heatmap(comparison, cmap="coolwarm", annot=False)
plt.show()
