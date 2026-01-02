import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs we care about: car=2, bus=5, truck=7
VEHICLE_CLASSES = [2, 5, 7]


def analyze_video(
    input_video_path: str,
    annotated_output_path: str,
    csv_writer,
    model,
    condition: str,
    model_name: str,
):
    os.makedirs(os.path.dirname(annotated_output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(annotated_output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]
        boxes = result.boxes

        num_detections = 0
        mean_confidence = 0.0

        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()

            mask = np.isin(cls, VEHICLE_CLASSES)
            vehicle_conf = conf[mask]

            if vehicle_conf.size > 0:
                num_detections = int(vehicle_conf.size)
                mean_confidence = float(vehicle_conf.mean())

        annotated_frame = result.plot()
        out.write(annotated_frame)

        csv_writer.writerow(
            [
                frame_idx,
                condition,
                model_name,
                num_detections,
                mean_confidence,
            ]
        )

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[{condition} | {model_name}] Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"[{condition} | {model_name}] Saved annotated video to: {annotated_output_path}")


def main():
    # 👇 choose which models you want to compare
    models_to_test = {
        "yolov8n": "yolov8n.pt",  # nano: fastest
        "yolov8s": "yolov8s.pt",  # small: better, a bit slower
        # you can add more:
        # "yolov8m": "yolov8m.pt",
    }

    clean_video = "data/raw/carss.mp4"
    degraded_video = "data/processed/carss_degraded.mp4"

    os.makedirs("results", exist_ok=True)
    csv_path = "results/models_comparison_stats.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "condition",      # "clean" or "degraded"
                "model",          # e.g. "yolov8n"
                "num_detections",
                "mean_confidence",
            ]
        )

        for model_name, weights in models_to_test.items():
            print(f"\n=== Loading {model_name} ({weights}) ===")
            model = YOLO(weights)

            # Clean
            analyze_video(
                input_video_path=clean_video,
                annotated_output_path=f"results/videos/carss_clean_{model_name}.mp4",
                csv_writer=writer,
                model=model,
                condition="clean",
                model_name=model_name,
            )

            # Degraded
            analyze_video(
                input_video_path=degraded_video,
                annotated_output_path=f"results/videos/carss_degraded_{model_name}.mp4",
                csv_writer=writer,
                model=model,
                condition="degraded",
                model_name=model_name,
            )

    print(f"\nSaved combined stats for all models to: {csv_path}")


if __name__ == "__main__":
    main()
