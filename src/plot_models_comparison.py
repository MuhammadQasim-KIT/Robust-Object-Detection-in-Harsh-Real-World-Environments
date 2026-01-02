import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_stats(csv_path: str):
    """
    Returns:
      stats[(condition, model)] = dict with lists:
        - "dets": list of num detections per frame
        - "confs": list of mean confidences per frame
    """
    stats = defaultdict(lambda: {"dets": [], "confs": []})

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["condition"]      # "clean" / "degraded"
            model = row["model"]             # "yolov8n" / "yolov8s"
            num_det = int(row["num_detections"])
            mean_conf = float(row["mean_confidence"])

            stats[(condition, model)]["dets"].append(num_det)
            stats[(condition, model)]["confs"].append(mean_conf)

    return stats


def main():
    os.makedirs("results/plots", exist_ok=True)
    csv_path = "results/models_comparison_stats.csv"

    stats = load_stats(csv_path)

    models = sorted({m for (_, m) in stats.keys()})
    conditions = ["clean", "degraded"]

    avg_dets = {cond: [] for cond in conditions}
    avg_confs = {cond: [] for cond in conditions}

    for cond in conditions:
        for model in models:
            key = (cond, model)
            if key in stats:
                dets = np.array(stats[key]["dets"])
                confs = np.array(stats[key]["confs"])

                avg_dets[cond].append(dets.mean())
                if np.any(confs > 0):
                    avg_confs[cond].append(confs[confs > 0].mean())
                else:
                    avg_confs[cond].append(0.0)
            else:
                avg_dets[cond].append(0.0)
                avg_confs[cond].append(0.0)

    x = np.arange(len(models))
    width = 0.35

    # 1) Average detections per frame
    plt.figure()
    plt.bar(x - width/2, avg_dets["clean"], width, label="Clean")
    plt.bar(x + width/2, avg_dets["degraded"], width, label="Degraded")
    plt.xticks(x, models)
    plt.ylabel("Avg. vehicle detections / frame")
    plt.title("Average detections per frame by model & condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/models_avg_detections.png", dpi=200)
    plt.close()

    # 2) Average mean confidence
    plt.figure()
    plt.bar(x - width/2, avg_confs["clean"], width, label="Clean")
    plt.bar(x + width/2, avg_confs["degraded"], width, label="Degraded")
    plt.xticks(x, models)
    plt.ylabel("Avg. mean confidence (vehicles)")
    plt.title("Average detection confidence by model & condition")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plots/models_avg_confidence.png", dpi=200)
    plt.close()

    print("Saved model comparison plots to results/plots/")


if __name__ == "__main__":
    main()
