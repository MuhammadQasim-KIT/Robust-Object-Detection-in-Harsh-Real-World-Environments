import cv2
from ultralytics import YOLO
import os


def main():
    # Paths
    input_video_path = "data/raw/carss.mp4"
    os.makedirs("results/videos", exist_ok=True)
    output_video_path = "results/videos/carss_yolo_baseline.mp4"

    # Load YOLOv8 model (pretrained COCO)
    model = YOLO("yolov8n.pt")  # 'n' = nano, fastest; good for laptops

    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        # YOLO drawing helper
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_video_path}")


if __name__ == "__main__":
    main()
