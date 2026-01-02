import cv2
import numpy as np
import os


def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_motion_blur(image, kernel_size=15):
    # Horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = 1.0
    kernel /= kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def add_fog(image, fog_intensity=0.5):
    # fog_intensity: 0 (none) to 1 (very strong)
    h, w, _ = image.shape
    fog_layer = np.full((h, w, 3), 255, dtype=np.uint8)
    foggy = cv2.addWeighted(image, 1 - fog_intensity, fog_layer, fog_intensity, 0)
    return foggy


def reduce_brightness(image, alpha=0.6, beta=-20):
    """
    alpha < 1.0 -> lower contrast
    beta < 0 -> darker
    """
    dark = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return dark


def add_dust_spots(image, num_spots=80, spot_radius=5):
    """
    Simulate dust as slightly brighter or darker blobs.
    Safer integer handling to avoid uint8 overflow errors.
    """
    h, w, _ = image.shape
    overlay = image.copy()

    for _ in range(num_spots):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(2, spot_radius)

        # Random brightness change
        color_shift = np.random.randint(-40, 40)

        # Safely convert each channel to int, add shift, clip, convert back
        b = int(np.clip(int(image[y, x, 0]) + color_shift, 0, 255))
        g = int(np.clip(int(image[y, x, 1]) + color_shift, 0, 255))
        r = int(np.clip(int(image[y, x, 2]) + color_shift, 0, 255))

        cv2.circle(overlay, (x, y), radius, (b, g, r), -1)

    dust_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return dust_image


def apply_harsh_conditions(frame):
    """
    Combine multiple effects to simulate harsh industrial conditions.
    Tune intensities here.
    """
    # 1) Reduce brightness / contrast
    frame = reduce_brightness(frame, alpha=0.7, beta=-30)

    # 2) Add fog/haze
    frame = add_fog(frame, fog_intensity=0.35)

    # 3) Add motion blur
    frame = add_motion_blur(frame, kernel_size=13)

    # 4) Add dust spots
    frame = add_dust_spots(frame, num_spots=60, spot_radius=7)

    # 5) Add sensor noise
    frame = add_gaussian_noise(frame, sigma=15)

    return frame


def main():
    input_video_path = "data/raw/carss.mp4"
    os.makedirs("data/processed", exist_ok=True)
    output_video_path = "data/processed/carss_degraded.mp4"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        degraded = apply_harsh_conditions(frame)
        out.write(degraded)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Saved degraded video to: {output_video_path}")


if __name__ == "__main__":
    main()
