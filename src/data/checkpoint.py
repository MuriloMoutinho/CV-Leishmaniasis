import hashlib, json
from dataclasses import asdict
from pathlib import Path

import torch

from handler import create_binary_model
from training import TrainingConfig

def config_tag(config: TrainingConfig, length: int = 8) -> str:
    params = asdict(config)
    params.pop("training_seed", None)
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:length]

def save_model(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(config: TrainingConfig, path: str | Path):
    path = Path(path)
    if not path.is_file():
        return None

    model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    print(f"Modelo carregado de {path}")
    return model
