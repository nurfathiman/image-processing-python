"""
Image Processing with Python - demo for ITT440 Assessment 1
Usage:
    python image_processing_demo.py                # uses a generated sample image
    python image_processing_demo.py my_photo.jpg   # uses your own image
Outputs are saved in the 'output/' folder.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path("output")
OUT.mkdir(exist_ok=True)


def make_sample_image(size=(400, 600)):
    """Create a colourful test image so the demo runs without any downloads."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # horizontal colour gradient
    for x in range(w):
        img[:, x] = (int(255 * x / w), 120, int(255 * (1 - x / w)))
    cv2.rectangle(img, (60, 60), (240, 220), (255, 255, 255), -1)
    cv2.circle(img, (430, 200), 90, (0, 220, 255), -1)
    cv2.putText(img, "ITT440", (90, 340), cv2.FONT_HERSHEY_SIMPLEX,
                2, (20, 20, 20), 4, cv2.LINE_AA)
    # add noise so the filters have something to do
    noise = np.random.default_rng(0).normal(0, 18, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def save(name, image):
    cv2.imwrite(str(OUT / f"{name}.png"), image)
    print(f"saved output/{name}.png")


def main():
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            sys.exit(f"Cannot read image: {sys.argv[1]}")
    else:
        img = make_sample_image()
    save("01_original", img)
    print("Shape (height, width, channels):", img.shape, "dtype:", img.dtype)

    # 1. Colour conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save("02_grayscale", gray)

    # 2. Geometric transformations
    resized = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    save("03_resized_half", resized)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    save("04_rotated_30deg", rotated)

    # 3. Noise removal / smoothing
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    save("05_gaussian_blur", blur)
    median = cv2.medianBlur(img, 5)
    save("06_median_filter", median)

    # 4. Contrast enhancement
    equalised = cv2.equalizeHist(gray)
    save("07_histogram_equalised", equalised)

    # 5. Thresholding (segmentation)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save("08_otsu_threshold", binary)

    # 6. Edge detection
    edges = cv2.Canny(blur, 80, 160)
    save("09_canny_edges", edges)

    # 7. Contours: count and outline shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = img.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)
    save("10_contours", outlined)
    print("Contours found:", len(contours))

    # 8. Side-by-side comparison for the video / article
    def to_bgr(a):
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a
    row1 = np.hstack([to_bgr(x) for x in (img, gray, blur)])
    row2 = np.hstack([to_bgr(x) for x in (equalised, binary, edges)])
    save("11_summary_grid", np.vstack([row1, row2]))


if __name__ == "__main__":
    main()
