from dataclasses import dataclass, asdict
import numpy

@dataclass
class TrainMetrics:
    loss: float
    accuracy: float

@dataclass
class ValidationMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    mcc: float
    precision: float
    recall: float
    specificity: float
    f1_class1: float
    f1_class0: float
    f1_macro: float
    roc_auc: float
    pr_auc_class1: float
    pr_auc_class0: float
    tn: int
    fp: int
    fn: int
    tp: int
    labels: numpy.ndarray
    probabilities: numpy.ndarray
    time: float = 0.0
    epoch: int | None = None

    ARRAY_FIELDS = ("labels", "probabilities")

    def get(self, name: str):
        if not hasattr(self, name) or name in self.ARRAY_FIELDS:
            raise KeyError(f"{name} não existe no retorno")
        return getattr(self, name)

    def scalars(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in self.ARRAY_FIELDS}
