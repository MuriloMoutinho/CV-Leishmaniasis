import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from datetime import datetime

from training import TrainingConfig, KFoldConfig, ValidationMetrics


PERCENT_METRICS = [
    "f1_class1", "f1_class0", "f1_macro", "f1_macro_tuned",
    "accuracy", "balanced_accuracy",
    "precision", "recall", "specificity",
    "roc_auc", "pr_auc_class1", "pr_auc_class0",
]
RAW_METRICS = ["mcc"]
CONFUSION_METRICS = ["tn", "fp", "fn", "tp"]


def _load_previous(filename):
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        return df_old, len(df_old) + 1
    return pd.DataFrame(), 1


def _aggregate_metrics(df: pd.DataFrame) -> dict:
    metrics = {}
    for name in PERCENT_METRICS:
        if name in df.columns:
            metrics[f"{name}_mean"] = round(df[name].mean() * 100, 2)
            metrics[f"{name}_std"] = round(df[name].std() * 100, 2)

    for name in RAW_METRICS:
        if name in df.columns:
            metrics[f"{name}_mean"] = round(df[name].mean(), 4)
            metrics[f"{name}_std"] = round(df[name].std(), 4)

    for name in CONFUSION_METRICS:
        if name in df.columns:
            metrics[f"{name}_sum"] = int(df[name].sum())
    return metrics


def _save_experiment(
    results: list[ValidationMetrics],
    config: TrainingConfig,
    info: dict,
    filename: str,
    extra=None,
    seeds: list[int] | None = None,
):
    df_old, experiment_id = _load_previous(filename)

    df_runs = pd.DataFrame([r.scalars() for r in results])

    result = {
        "experiment_id": experiment_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time": round(df_runs["time"].sum() / 60, 2),
        **info,
        **asdict(config),
        **_aggregate_metrics(df_runs),
    }
    if seeds is not None:
        result["training_seed"] = ",".join(str(s) for s in seeds)
    if extra is not None:
        result.update(extra(df_runs))

    df_result = pd.concat([df_old, pd.DataFrame([result])], ignore_index=True)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")

    return experiment_id


def save_kfold_result(
    fold_results: list[ValidationMetrics],
    config: TrainingConfig,
    dataset_name: str,
    kfold_config: KFoldConfig,
    filename="kfold.csv"
):
    def extra(df_folds):
        values = {
            "best_epoch_mean": round(df_folds["epoch"].mean(), 2),
            "best_epoch_median": round(df_folds["epoch"].median(), 2),
            "best_epoch_std": round(df_folds["epoch"].std(), 2),
        }
        if kfold_config is not None:
            values.update(asdict(kfold_config))
        return values

    return _save_experiment(
        fold_results, config,
        info={"dataset_name": dataset_name},
        filename=filename,
        extra=extra,
    )


def save_training_result(
    model_result: list[ValidationMetrics],
    config: TrainingConfig,
    dataset_name: str,
    filename="training.csv",
    seeds: list[int] | None = None
):
    return _save_experiment(
        model_result, config,
        info={"dataset_name": dataset_name},
        filename=filename,
        seeds=seeds,
    )


def save_cross_dataset_result(
    model_result: list[ValidationMetrics],
    trained_on: str,
    tested_on: str,
    config: TrainingConfig,
    filename="cross_dataset_results.csv",
    seeds: list[int] | None = None
):
    return _save_experiment(
        model_result, config,
        info={"trained_on": trained_on, "tested_on": tested_on},
        filename=filename,
        seeds=seeds,
    )
