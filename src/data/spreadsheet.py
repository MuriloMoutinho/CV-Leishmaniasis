import os, numpy
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from datetime import datetime

from training import TrainingConfig, KFoldConfig, HoldoutConfig


def save_kfold_result(
    fold_results,
    config: TrainingConfig,
    dataset_name: str,
    kfold_config: KFoldConfig,
    filename="kfold.csv"
):
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        experiment_id = len(df_old) + 1
    else:
        df_old = pd.DataFrame()
        experiment_id = 1

    experiment_dir = Path(filename).parent / "labels" / str(experiment_id)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    for fold, fold_result in enumerate(fold_results, start=1):
        numpy.savez(
            experiment_dir / f"fold{fold}.npz",
            labels=fold_result["labels"],
            probabilities=fold_result["probabilities"],
        )

    scalar_results = [
        {k: v for k, v in r.items() if k not in ("labels", "probabilities")}
        for r in fold_results
    ]
    df_folds = pd.DataFrame(scalar_results)

    result = {
        "experiment_id": experiment_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time": round(df_folds["time"].sum() / 60, 2),
        "dataset_name": dataset_name,
    }

    result.update(asdict(config))

    percent_metrics = [
        "f1", "f1_class0", "f1_macro", "f1_macro_tuned",
        "accuracy", "balanced_accuracy",
        "precision", "recall",
        "roc_auc", "pr_auc_class1", "pr_auc_class0",
    ]
    for name in percent_metrics:
        if name in df_folds.columns:
            result[f"{name}_mean"] = round(df_folds[name].mean() * 100, 2)
            result[f"{name}_std"] = round(df_folds[name].std() * 100, 2)

    threshold_metrics = ["mcc"]
    for name in threshold_metrics:
        if name in df_folds.columns:
            result[f"{name}_mean"] = round(df_folds[name].mean(), 4)
            result[f"{name}_std"] = round(df_folds[name].std(), 4)

    for name in ["tn", "fp", "fn", "tp"]:
        if name in df_folds.columns:
            result[f"{name}_sum"] = int(df_folds[name].sum())

    result.update({
        "best_epoch_mean": round(df_folds["epoch"].mean(), 2),
        "best_epoch_median": round(df_folds["epoch"].median(), 2),
        "best_epoch_std": round(df_folds["epoch"].std(), 2),
    })

    if kfold_config is not None:
        result.update(asdict(kfold_config))

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")

    return df_result

def save_holdout_result(
    model_result,
    config: TrainingConfig,
    holdout_config: HoldoutConfig,
    dataset_name: str,
    filename="holdout.csv"
):
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        experiment_id = len(df_old) + 1
    else:
        df_old = pd.DataFrame()
        experiment_id = 1

    result = {
        "experiment_id": experiment_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time": round(model_result['time'] / 60, 2),
        "dataset_name": dataset_name,
    }

    result.update(asdict(config))

    result.update({
        "f1": round(model_result['f1'] * 100, 2),
        "accuracy": round(model_result['accuracy'] * 100, 2),
        "precision": round(model_result['precision'] * 100, 2),
        "recall": round(model_result['recall'] * 100, 2),
        "roc_auc": round(model_result['roc_auc'] * 100, 2),
        "pr_auc": round(model_result['pr_auc'] * 100, 2),
        "best_epoch": round(model_result["epoch"], 2),
    })

    result.update(asdict(holdout_config))

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")
    return df_result


def save_training_result(
    model_result,
    config: TrainingConfig,
    dataset_name: str,
    filename="training.csv"
):
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        experiment_id = len(df_old) + 1
    else:
        df_old = pd.DataFrame()
        experiment_id = 1

    result = {
        "experiment_id": experiment_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time":  round(model_result['time'] / 60, 2),
        "dataset_name": dataset_name,
    }

    result.update(asdict(config))

    result.update({
        "f1": round(model_result['f1'] * 100, 2),
        "accuracy": round(model_result['accuracy'] * 100, 2),
        "precision": round(model_result['precision'] * 100, 2),
        "recall": round(model_result['recall'] * 100, 2),
        "roc_auc": round(model_result['roc_auc'] * 100, 2),
        "pr_auc": round(model_result['pr_auc'] * 100, 2),
    })

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")
    return df_result


def save_cross_dataset_result(
    model_result,
    trained_on: str,
    tested_on: str,
    config: TrainingConfig,
    filename="cross_dataset_results.csv"
):
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        experiment_id = len(df_old) + 1
    else:
        df_old = pd.DataFrame()
        experiment_id = 1

    result = {
        "experiment_id": experiment_id,
        "time": round(model_result.get('time', 0) / 60, 2),
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trained_on": trained_on,
        "tested_on": tested_on,
        "model_name": config.model_name,
        "fine_tuning": config.fine_tuning,
        "augmentation_level": config.augmentation_level,
    }

    result.update({
        "f1": round(model_result['f1'] * 100, 2),
        "accuracy": round(model_result['accuracy'] * 100, 2),
        "precision": round(model_result['precision'] * 100, 2),
        "recall": round(model_result['recall'] * 100, 2),
        "roc_auc": round(model_result['roc_auc'] * 100, 2),
        "pr_auc": round(model_result['pr_auc'] * 100, 2),
    })

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo ({trained_on} -> {tested_on})")
    return df_result