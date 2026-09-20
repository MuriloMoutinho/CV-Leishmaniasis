import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from datetime import datetime

from config import TrainingConfig, KFoldConfig, HoldoutConfig


def save_kfold_result(
    fold_results,
    config: TrainingConfig,
    dataset_name: str,
    kfold_config: KFoldConfig,
    filename="kfold.csv"
):
    df_folds = pd.DataFrame(fold_results)

    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        experiment_id = len(df_old) + 1
    else:
        df_old = pd.DataFrame()
        experiment_id = 1

    result = {
        "experiment_id": experiment_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time": round(df_folds["time"].sum() / 60, 2),
        "dataset_name": dataset_name,
    }

    result.update(asdict(config))

    result.update({
        "f1_mean":  round(df_folds["f1"].mean() * 100, 2),
        "f1_std":  round(df_folds["f1"].std() * 100, 2),

        "accuracy_mean":  round(df_folds["accuracy"].mean() * 100, 2),
        "accuracy_std":  round(df_folds["accuracy"].std() * 100, 2),

        "precision_mean":  round(df_folds["precision"].mean() * 100, 2),
        "precision_std":  round(df_folds["precision"].std() * 100, 2),

        "recall_mean":  round(df_folds["recall"].mean() * 100, 2),
        "recall_std":  round(df_folds["recall"].std() * 100, 2),

        "roc_auc_mean":  round(df_folds["roc_auc"].mean() * 100, 2),
        "roc_auc_std":  round(df_folds["roc_auc"].std() * 100, 2),

        "pr_auc_mean": round(df_folds['pr_auc'].mean() * 100, 2),
        "pr_auc_std": round(df_folds['pr_auc'].std() * 100, 2),

        "best_epoch_mean":  round(df_folds["epoch"].mean(), 2),
        "best_epoch_median":  round(df_folds["epoch"].median(), 2),
        "best_epoch_std":  round(df_folds["epoch"].std(), 2),
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
    