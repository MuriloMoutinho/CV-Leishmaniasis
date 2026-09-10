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

    result = asdict(config)

    result.update({
        "experiment_id": experiment_id,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "accuracy_mean": df_folds["accuracy"].mean(),
        "accuracy_std": df_folds["accuracy"].std(),

        "precision_mean": df_folds["precision"].mean(),
        "precision_std": df_folds["precision"].std(),

        "recall_mean": df_folds["recall"].mean(),
        "recall_std": df_folds["recall"].std(),

        "f1_mean": df_folds["f1"].mean(),
        "f1_std": df_folds["f1"].std(),

        "auc_mean": df_folds["auc"].mean(),
        "auc_std": df_folds["auc"].std(),

        "epoch_mean": df_folds["epoch"].mean(),
        "epoch_std": df_folds["epoch"].std(),
    })

    result.update(asdict(kfold_config))

    df_result = pd.DataFrame([result])
    df_result = pd.concat([df_old, df_result], ignore_index=True)

    df_result.to_csv(filename, index=False)
    print(f"\nExperimento {experiment_id} salvo")
    return df_result
