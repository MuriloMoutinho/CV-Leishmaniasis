from pathlib import Path

import numpy


def save_npz(results, output_dir: str | Path, file_prefix: str = "model"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for n, res in enumerate(results, start=1):
        numpy.savez(
            output_dir / f"{file_prefix}{n}.npz",
            labels=res["labels"],
            probabilities=res["probabilities"],
        )
