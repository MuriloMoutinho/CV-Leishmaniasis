import os
from pathlib import Path
import gc, torch, optuna, torchvision, torch.cuda, ast
import pandas as pd
from dataclasses import replace

from torch.utils.data import ConcatDataset

from data import save_kfold_result, save_npz
from training import train_validate_kfold, KFoldConfig, TrainingConfig

DATASET_ROOT = Path("../datasets-folds")
DATASET_CROSS_ROOT = Path("../datasets-cross")

DATASETS = {
    "CROSS_AIR_LEISH": DATASET_CROSS_ROOT / "AIR_LEISH",
    "CROSS_DeepLeish": DATASET_CROSS_ROOT / "DeepLeish",
    "CROSS_DLB": DATASET_CROSS_ROOT / "DLB",

    "DLB+Deep": ["CROSS_DLB", "CROSS_DeepLeish"],
    "AIR+Deep": ["CROSS_AIR_LEISH", "CROSS_DeepLeish"],
    "DLB+AIR": ["CROSS_DLB", "CROSS_AIR_LEISH"],

    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH" / "train",
    "DeepLeish": DATASET_ROOT / "DeepLeish" / "train",
    "DLB": DATASET_ROOT / "DLB" / "train",
}

N_TRIALS = 15
BATCH_SIZE = 24
RANDOM_SEED = 42

EXPERIMENTS_ROOT = "experiments/loso2/pre-results-loso/optuna"


def load_dataset(name: str):
    root = DATASETS[name]

    if isinstance(root, list):

        datasets = [
            torchvision.datasets.ImageFolder(root=str(DATASETS[dataset_name]))
            for dataset_name in root
        ]

        reference_classes = datasets[0].class_to_idx

        for ds in datasets[1:]:
            if ds.class_to_idx != reference_classes:
                raise ValueError(f"Classes incompatíveis entre os datasets: {reference_classes} != {ds.class_to_idx}")

        return ConcatDataset(datasets)

    return torchvision.datasets.ImageFolder(root=str(root))

kfold_config = KFoldConfig(
    n_splits=3,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="mcc",
)

SEARCH_SPACE = {
    "learning_rate": [1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "resolution": [(224, 168), (384, 288), (576, 432)],
    "fine_tuning": [None, "last_block"] #"last_two_blocks"
}

MODEL_NAME = [
    "resnet50",
    "densenet121",
    "efficientnetb0",
]

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", training_seed=42,
                                epochs=100, scheduler_name=None, augmentation_level="weak",
                                model_name="", dropout=0, learning_rate=0, weight_decay=0, batch_size=0)

def suggest_config(trial, model_name):

    learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"])
    weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"])
    dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
    fine_tuning = trial.suggest_categorical("fine_tuning", SEARCH_SPACE["fine_tuning"])

    resolution = trial.suggest_categorical("resolution",
                                           [str(resolution) for resolution in SEARCH_SPACE["resolution"]])
    resolution = ast.literal_eval(resolution)

    config = replace(
        default_config,
        model_name=model_name,
        fine_tuning=fine_tuning,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_size=BATCH_SIZE,
        resolution=resolution,
    )

    return config

def evaluate_config_on_dataset(config, dataset_name, model_name, trial):
    dataset = load_dataset(dataset_name)

    fold_results = train_validate_kfold(
        dataset,
        config=config,
        kfold_config=kfold_config,
        trial=trial
    )

    experiment_id = save_kfold_result(
        fold_results=fold_results,
        config=config,
        dataset_name=dataset_name,
        kfold_config=kfold_config,
        filename=f"{EXPERIMENTS_ROOT}/kfold/optuna_kfold_{model_name}_{dataset_name}.csv",
    )
    save_npz(fold_results, f"{EXPERIMENTS_ROOT}/kfold/labels-kfold/{model_name}_{dataset_name}/{experiment_id}", file_prefix="fold")

    df_folds = pd.DataFrame([r.scalars() for r in fold_results])
    score = df_folds[kfold_config.metric_to_monitor].mean()

    torch.cuda.empty_cache()
    gc.collect()

    return score

def objective(trial, model_name, dataset_name):
    config = suggest_config(
        trial=trial,
        model_name=model_name,
    )

    print("\n" + "-" * 70)
    print(f"TRIAL {trial.number}")
    print(f"Model={model_name} | Dataset={dataset_name}")
    print(
        f"Fine-tuning={config.fine_tuning} | "
        f"LR={config.learning_rate:.6g} | "
        f"WD={config.weight_decay:.6g} | "
        f"Dropout={config.dropout} | "
        f"Resolution={config.resolution}"
    )
    print("-" * 70)

    score = evaluate_config_on_dataset(
        config=config,
        dataset_name=dataset_name,
        model_name=model_name,
        trial=trial
    )

    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("dataset_name", dataset_name)
    trial.set_user_attr("batch_size", config.batch_size)

    print(f"\n{kfold_config.metric_to_monitor} ({dataset_name}) = {score:.4f}")

    return score

def save_optuna_results(study, model_name, dataset_name):
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    path = (f"{EXPERIMENTS_ROOT}/optuna_{model_name}_{dataset_name}.csv")
    df.to_csv(path, index=False)

    return df


def run_optuna(model_name, dataset_name, trials):
    study_name = f"{model_name}_{dataset_name}"

    print("\n" + "=" * 70)
    print(f"OPTUNA: {model_name} | DATASET: {dataset_name}")
    print("=" * 70)

    os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{EXPERIMENTS_ROOT}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_name=model_name,
            dataset_name=dataset_name,
        ),
        n_trials=trials,
    )

    print("\n" + "=" * 70)
    print(f"MELHOR CONFIGURAÇÃO: {study_name}")
    print(f"Trial: {study.best_trial.number}")
    print(f"{kfold_config.metric_to_monitor}: {study.best_value:.4f}")

    print("\nParâmetros:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    print("\nAtributos:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")

    print("=" * 70)

    save_optuna_results(
        study=study,
        model_name=model_name,
        dataset_name=dataset_name,
    )

    return study

if __name__ == "__main__":
    for model_name in MODEL_NAME:
        for dataset_name in DATASETS.keys():
            study = run_optuna(
                model_name=model_name,
                dataset_name=dataset_name,
                trials=N_TRIALS,
            )