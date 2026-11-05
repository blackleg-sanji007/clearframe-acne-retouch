"""Render the recorded validation curves without loading any model or dataset."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "reports/training.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = json.loads((ROOT / "models/acne.json").read_text(encoding="utf-8"))["best_epoch"]
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained")
    for axis, key, title, color in [
        (axes[0], "metrics/mAP50(B)", "Validation mAP@0.50", "#087f8c"),
        (axes[1], "metrics/mAP50-95(B)", "Validation mAP@0.50:0.95", "#334e68"),
    ]:
        scores = [float(row[key]) for row in rows]
        axis.plot(epochs, scores, color=color, linewidth=1.8)
        axis.axvline(selected, color="#b45309", linestyle="--", linewidth=1.2,
                     label=f"Selected checkpoint: epoch {selected}")
        axis.scatter([selected], [scores[epochs.index(selected)]], color="#b45309", zorder=3)
        axis.set(title=title, xlabel="Completed epoch", ylabel="Score", xlim=(1, max(epochs)), ylim=(0, None))
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("Clearframe · Recorded ACNE04 detector training", fontweight="bold")
    target = ROOT / "docs/assets/validation-curves.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
