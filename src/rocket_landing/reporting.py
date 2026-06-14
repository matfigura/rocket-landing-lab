import json
from dataclasses import asdict
from pathlib import Path

from rocket_landing.models import EvaluationSummary


def save_evaluation_summary(
    summary: EvaluationSummary,
    output_path: str | Path,
) -> Path:
    """Zapisuje podsumowanie ewaluacji do pliku JSON."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_data = asdict(summary)

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_data,
            file,
            indent=4,
        )
        file.write("\n")

    return path


def load_evaluation_summary(
    input_path: str | Path,
) -> EvaluationSummary:
    """Odczytuje podsumowanie ewaluacji z pliku JSON."""

    path = Path(input_path)

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        summary_data = json.load(file)

    return EvaluationSummary(**summary_data)