import wandb
from ails_miccai_uwf4dr_challenge.config import WANDB_API_KEY
from train_model_plain import train

# Define the sweep configuration
sweep_config = {
    "method": "random",  # or "grid", or "bayes"
    "parameters": {
        "learning_rate": {
            "values": [1e-4, 1e-3, 1e-2]
        },
        "epochs": {
            "values": [10, 15, 20]
        },
        "batch_size": {
            "values": [4, 8, 9]
        },
        "model_type": {
            "values": ["AutoMorphModel", "Task1EfficientNetB4"]
        }
    }
}

wandb.login(key=WANDB_API_KEY)

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="task1")

# Start the sweep
wandb.agent(sweep_id, function=train)