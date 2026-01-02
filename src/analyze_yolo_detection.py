import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# COCO class IDs we care about: car=2, bus=5, truck=7
VEHICLE_CLASSES = [2, 5, 7]


def analyze_video(input_video_path, annotated_output_path, csv_output_path, model, label=""):
    os.makedirs(os.path.dirname(annotated_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(annotated_output_path, fourcc, fps, (width, height))

    # CSV writer
    with open(csv_output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "label", "num_detections", "mean_confidence"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference
            results = model(frame, verbose=False)
            result = results[0]
            boxes = result.boxes

            num_detections = 0
            mean_confidence = 0.0

            if boxes is not None and len(boxes) > 0:
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()

                # keep only vehicles (car, bus, truck)
                mask = np.isin(cls, VEHICLE_CLASSES)
                vehicle_conf = conf[mask]

                if vehicle_conf.size > 0:
                    num_detections = int(vehicle_conf.size)
                    mean_confidence = float(vehicle_conf.mean())

            # Draw annotations on frame
            annotated_frame = result.plot()

            out.write(annotated_frame)

            # Write stats
            writer.writerow([frame_idx, label, num_detections, mean_confidence])

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[{label}] Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"[{label}] Saved annotated video to: {annotated_output_path}")
    print(f"[{label}] Saved stats CSV to: {csv_output_path}")


def main():
    model = YOLO("yolov8n.pt")  # nano model, fast

    # 1) Clean video
    analyze_video(
        input_video_path="data/raw/carss.mp4",
        annotated_output_path="results/videos/carss_clean_yolo_annotated.mp4",
        csv_output_path="results/carss_clean_stats.csv",
        model=model,
        label="clean",
    )

    # 2) Degraded video
    analyze_video(
        input_video_path="data/processed/carss_degraded.mp4",
        annotated_output_path="results/videos/carss_degraded_yolo_annotated.mp4",
        csv_output_path="results/carss_degraded_stats.csv",
        model=model,
        label="degraded",
    )


if __name__ == "__main__":
    main()
