import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_stats(csv_path):
    frames = []
    num_detections = []
    mean_conf = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_idx"]))
            num_detections.append(int(row["num_detections"]))
            mean_conf.append(float(row["mean_confidence"]))

    return np.array(frames), np.array(num_detections), np.array(mean_conf)


def main():
    os.makedirs("results/plots", exist_ok=True)

    clean_csv = "results/carss_clean_stats.csv"
    degraded_csv = "results/carss_degraded_stats.csv"

    frames_clean, dets_clean, conf_clean = load_stats(clean_csv)
    frames_deg, dets_deg, conf_deg = load_stats(degraded_csv)

    # 1) Line plot: detections per frame
    plt.figure()
    plt.plot(frames_clean, dets_clean, label="Clean", linewidth=1.5)
    plt.plot(frames_deg, dets_deg, label="Degraded", linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Vehicle detections (car/bus/truck)")
    plt.title("Detections per frame: Clean vs Degraded")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/carss_detections_vs_frame.png", dpi=200)
    plt.close()

    # 2) Line plot: mean confidence per frame
    plt.figure()
    plt.plot(frames_clean, conf_clean, label="Clean", linewidth=1.5)
    plt.plot(frames_deg, conf_deg, label="Degraded", linewidth=1.5)
    plt.xlabel("Frame")
    plt.ylabel("Mean confidence (vehicles)")
    plt.title("Mean detection confidence: Clean vs Degraded")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/carss_confidence_vs_frame.png", dpi=200)
    plt.close()

    # 3) Bar plot: global averages
    avg_det_clean = dets_clean.mean()
    avg_det_deg = dets_deg.mean()
    avg_conf_clean = conf_clean[conf_clean > 0].mean() if np.any(conf_clean > 0) else 0.0
    avg_conf_deg = conf_deg[conf_deg > 0].mean() if np.any(conf_deg > 0) else 0.0

    # Bar chart for detections
    plt.figure()
    plt.bar(["Clean", "Degraded"], [avg_det_clean, avg_det_deg])
    plt.ylabel("Avg. vehicle detections per frame")
    plt.title("Average detections: Clean vs Degraded")
    plt.tight_layout()
    plt.savefig("results/plots/carss_avg_detections_bar.png", dpi=200)
    plt.close()

    # Bar chart for confidence
    plt.figure()
    plt.bar(["Clean", "Degraded"], [avg_conf_clean, avg_conf_deg])
    plt.ylabel("Avg. mean confidence (vehicles)")
    plt.title("Average confidence: Clean vs Degraded")
    plt.tight_layout()
    plt.savefig("results/plots/carss_avg_confidence_bar.png", dpi=200)
    plt.close()

    print("Saved plots to results/plots/")


if __name__ == "__main__":
    main()
