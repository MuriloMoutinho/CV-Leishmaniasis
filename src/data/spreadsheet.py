import os
from dataclasses import asdict

import pandas as pd
from datetime import datetime

from config import TrainingConfig, KFoldConfig


def save_experiment_result(
    fold_results,
    config: TrainingConfig,
    kfold_config: KFoldConfig,
    filename="experimentos.csv"
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
        "time": df_folds["time"].sum() / 60,
    }

    result.update(asdict(config))

    result.update({
        "accuracy_mean": df_folds["accuracy"].mean() * 100,
        "accuracy_std": df_folds["accuracy"].std() * 100,

        "precision_mean": df_folds["precision"].mean() * 100,
        "precision_std": df_folds["precision"].std() * 100,

        "recall_mean": df_folds["recall"].mean() * 100,
        "recall_std": df_folds["recall"].std() * 100,

        "f1_mean": df_folds["f1"].mean() * 100,
        "f1_std": df_folds["f1"].std() * 100,

        "auc_mean": df_folds["auc"].mean() * 100,
        "auc_std": df_folds["auc"].std() * 100,

        "epoch_mean": df_folds["epoch"].mean(),
        "epoch_median": df_folds["epoch"].median(),
        "epoch_std": df_folds["epoch"].std(),
    })

    if kfold_config is not None:
        result.update(asdict(kfold_config))

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")
    return df_result


def save_training_result(
    model_result,
    config: TrainingConfig,
    filename="experimentos.csv"
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
        "time": model_result['time'] / 60,
    }

    result.update(asdict(config))

    result.update({
        "accuracy": model_result['accuracy'] * 100,
        "precision": model_result['precision'] * 100,
        "recall": model_result['recall'] * 100,
        "f1": model_result['f1'] * 100,
        "auc": model_result['auc'] * 100,
    })

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")
    return df_result
