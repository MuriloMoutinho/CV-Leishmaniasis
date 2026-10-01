import os
from pathlib import Path
import gc, torch, optuna, torchvision, torch.cuda, ast
import pandas as pd
from dataclasses import replace
from training_config import KFoldConfig, TrainingConfig
from data import save_kfold_result
from training import train_validate_kfold

DATASET_ROOT = Path("../datasets-folds")

DATASETS = {
    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH" / "train",
    "DeepLeish": DATASET_ROOT / "DeepLeish" / "train",
    "DLB": DATASET_ROOT / "DLB" / "train",
}

N_TRIALS = 5
RANDOM_SEED = 42

def load_dataset(name: str):
    root = DATASETS[name]
    return torchvision.datasets.ImageFolder(root=str(root))

kfold_config = KFoldConfig(
    n_splits=3,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="f1",
)

SEARCH_SPACE = {
    "learning_rate": [1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "resolution": [(224, 168), (384, 288), (576, 432), (768, 576)],
    "fine_tuning": [None, "last_block", "last_two_blocks"]
}

MODEL_NAME = [
    "resnet50",
    "densenet121",
    "efficientnetb0",
]

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw",
                                epochs=50, scheduler_name=None, augmentation_level="weak",
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
        batch_size=16,
        resolution=resolution,
    )

    return config

def evaluate_config_on_datasets(config):
    dataset_scores = {}

    for dataset_name in DATASETS.keys():
        dataset = load_dataset(dataset_name)

        fold_results, fold_histories = train_validate_kfold(
            dataset,
            config=config,
            kfold_config=kfold_config,
        )

        save_kfold_result(
            fold_results=fold_results,
            config=config,
            dataset_name=dataset_name,
            kfold_config=kfold_config,
            filename="experiments/optuna_kfold.csv"
        )

        df_folds = pd.DataFrame(fold_results)
        dataset_scores[dataset_name] = df_folds["f1"].mean()

        torch.cuda.empty_cache()
        gc.collect()

    global_score = sum(dataset_scores.values()) / len(dataset_scores)

    return global_score, dataset_scores

def objective(trial, model_name):
    config = suggest_config(
        trial=trial,
        model_name=model_name,
    )

    print("\n" + "-" * 70)
    print(f"TRIAL {trial.number}")
    print(
        f"Model={model_name} | Fine-tuning={config.fine_tuning} | "
        f"LR={config.learning_rate:.6g} | "
        f"WD={config.weight_decay:.6g} | "
        f"Dropout={config.dropout} | "
        f"Resolution={config.resolution}"
    )
    print("-" * 70)

    global_score, dataset_scores = evaluate_config_on_datasets(config=config)

    for dataset_name, score in dataset_scores.items():
        trial.set_user_attr(f"f1_{dataset_name}", score)

    trial.set_user_attr("model_name", model_name)
    trial.set_user_attr("batch_size", config.batch_size)

    print(f"\nF1 DLB        = {dataset_scores['DLB']:.4f}")
    print(f"F1 AIR_LEISH  = {dataset_scores['AIR_LEISH']:.4f}")
    print(f"F1 DeepLeish  = {dataset_scores['DeepLeish']:.4f}")
    print(f"F1 médio      = {global_score:.4f}")

    return global_score

def save_optuna_results(study, model_name):
    os.makedirs("experiments", exist_ok=True)

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    path = f"experiments/optuna_{model_name}.csv"
    df.to_csv(path, index=False)

    return df


def run_optuna(model_name, trials):
    print("\n" + "=" * 70)
    print(f"OPTUNA: {model_name}")
    print("=" * 70)

    study_name = f"{model_name}"

    os.makedirs("experiments", exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",

        storage="sqlite:///experiments/optuna.db",
        load_if_exists=True,

        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    study.optimize(
        lambda trial: objective(trial, model_name=model_name),
        n_trials=trials,
    )

    print("\n" + "=" * 70)
    print("MELHOR CONFIGURAÇÃO")
    print(f"Trial: {study.best_trial.number}")
    print(f"F1 médio: {study.best_value:.4f}")
    print("\nParâmetros:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print("\nAtributos:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    for dataset_name in DATASETS.keys():
        value = study.best_trial.user_attrs.get(f"f1_{dataset_name}")
        print(f"  {dataset_name}: {value:.4f}")

    save_optuna_results(study=study, model_name=model_name)

    return study

if __name__ == "__main__":
    for model_name in MODEL_NAME:
        run_optuna(model_name=model_name, trials=N_TRIALS)