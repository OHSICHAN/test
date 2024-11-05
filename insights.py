# %%
import torch
import os
import pandas as pd
import torch
from IPython.display import display


PROJECT_PATH = "/Users/saankim/Library/CloudStorage/CloudMounter-bio/dev/DeepInteract/"

# Load the saved model weights
saved_model_path = PROJECT_PATH + "deepinteratct2/"
# Iterate over every .pth file in the path
PROJECT_PATH = "/Users/saankim/Library/CloudStorage/CloudMounter-bio/dev/DeepInteract/"

# Load the saved model weights
saved_model_path = PROJECT_PATH + "deepinteratct2/"
# Iterate over every .pth file in the path
for file_name in os.listdir(saved_model_path):
    if file_name.endswith(".pth"):
        file_path = os.path.join(saved_model_path, file_name)
        model_state_dict = torch.load(
            file_path, map_location=torch.device("cpu"), weights_only=True
        )

        # Create a dataframe to store the model weights
        data = []
        columns = ["Parameter", "Value"]

        # Print the model weights
        param_names = {
            # "scale": "protein_projection.soft{}.attention.scale",
            # "width": "protein_projection.soft{}.attention.width",
            "residual": "protein_projection.soft{}.residual"
        }

        for i in range(1, 6):  # soft1 to soft5
            for key, value in param_names.items():
                param_name = value.format(i)
                if param_name in model_state_dict:
                    data.append([param_name, model_state_dict[param_name]])
                else:
                    data.append([param_name, "Not found in the state_dict"])

        # Create a dataframe and display as a table
        df = pd.DataFrame(data, columns=columns)
        display(df)
