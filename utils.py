import torch
from torch import nn
from pathlib import Path

def save_model(model: nn.Module,
               optimizer: torch.optim,
               target_dir: str,
               model_name: str)->None:
    
    # Create target directory
    target_dir_path = Path(target_dir)

    if not target_dir_path.is_dir():
        target_dir_path.mkdir(parents=True,exist_ok=True)

    # Create model saving path
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), "Model name should be a .pt ot .pth extension"
    model_save_path = target_dir_path / model_name

    # Save Required params
    print(f"[INFO] Saving model to: {model_save_path}")
    context = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(context, model_save_path)