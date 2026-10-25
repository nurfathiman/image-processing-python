# Image Processing with Python

**Course:** ITT440 | **Assessment 1 (Individual Assignment)**
**Name:** MUHAMMAD NURFATH IMAN BIN ROOSEFFENDY | **Student ID:** 2025183455

> 🎥 **Demo video:** [https://youtu.be/nnKj0J-UlEY](https://youtu.be/nnKj0J-UlEY)

---

## 1. Introduction

Every photo on your phone, every X-ray in a hospital and every frame a self-driving car sees is a grid of numbers. **Image processing** is the practice of applying mathematical operations to those numbers to improve an image or extract useful information from it. Typical goals include removing noise, enhancing contrast, finding edges, separating objects from the background, and preparing images for machine learning.

Python has become one of the most popular languages for this work. It is easy to read, and it has mature libraries that do the heavy lifting in fast, compiled code. This article introduces the core ideas and walks through a working example using **OpenCV** and **NumPy**. A demo video is linked above.

## 2. How a digital image is represented

A digital image is a matrix of **pixels**.

| Image type | Shape in NumPy | Pixel value |
|---|---|---|
| Grayscale | `(height, width)` | One number, 0 (black) to 255 (white) |
| Colour | `(height, width, 3)` | Three numbers (B, G, R in OpenCV) |

Because an image is just a NumPy array, any array operation (slicing, arithmetic, masking) is also an image operation. One thing to remember: OpenCV loads colour images in **BGR** order, not RGB.

## 3. Popular Python libraries

| Library | Best used for |
|---|---|
| **OpenCV** (`cv2`) | Fast, general-purpose computer vision: filtering, edges, geometry, video [1] |
| **Pillow** (`PIL`) | Simple tasks: open, crop, resize, convert, save many formats [2] |
| **scikit-image** | Scientific image analysis with a clean NumPy-based API [3] |
| **NumPy** | The array foundation that all the libraries above use [4] |
| **Matplotlib** | Displaying images and plots [5] |

This article uses OpenCV and NumPy.

## 4. Key techniques

1. **Colour space conversion.** Converting BGR to grayscale reduces three channels to one, which simplifies and speeds up later steps.
2. **Geometric transformations.** Resizing and rotating change the position or size of pixels. Rotation uses an affine transformation matrix.
3. **Smoothing and noise removal.** A *Gaussian blur* replaces each pixel with a weighted average of its neighbours. A *median filter* uses the median instead and is especially good against "salt-and-pepper" noise.
4. **Histogram equalisation.** Spreads out the brightness values to increase contrast in dull images.
5. **Thresholding.** Turns a grayscale image into black and white. *Otsu's method* chooses the threshold automatically by analysing the histogram [6].
6. **Edge detection.** The *Canny* detector finds sharp changes in intensity, which usually correspond to object boundaries [7].
7. **Contour detection.** Traces the outlines of shapes found in a binary or edge image, so objects can be counted and measured.

## 5. Programming example

### Setup

```bash
git clone https://github.com/nurfathiman/image-processing-python.git
cd image-processing-python
pip install -r requirements.txt
```

### Run

```bash
python image_processing_demo.py              # uses a generated sample image
python image_processing_demo.py my_photo.jpg # uses your own image
```

Results are saved in the `output/` folder.

### Code walkthrough

The full script is in [`image_processing_demo.py`](image_processing_demo.py). The key steps:

```python
import cv2

img = cv2.imread("my_photo.jpg")                        # load (BGR array)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # 1. grayscale

h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)    # 2. rotate 30 degrees
rotated = cv2.warpAffine(img, M, (w, h))

blur = cv2.GaussianBlur(img, (7, 7), 0)                 # 3. smoothing
median = cv2.medianBlur(img, 5)

equalised = cv2.equalizeHist(gray)                      # 4. contrast

_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 5. Otsu

edges = cv2.Canny(blur, 80, 160)                        # 6. edges

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE) # 7. contours
print("Contours found:", len(contours))
```

### Results

The grid below shows the sample image after each step. Top row: original, grayscale, Gaussian blur. Bottom row: histogram equalisation, Otsu threshold, Canny edges.

![Summary grid](images/11_summary_grid.png)

| Step | Output |
|---|---|
| Original | ![original](images/01_original.png) |
| Rotated 30° | ![rotated](images/04_rotated_30deg.png) |
| Contours drawn in green | ![contours](images/10_contours.png) |

Observations:

- Gaussian blur softens the noise but also softens edges. A trade-off to keep in mind.
- Otsu's threshold cleanly separates the bright square and circle from the darker background.
- Canny finds the outlines of the square, circle and text, and runs best on a blurred image, because noise otherwise creates false edges.

## 6. Real-world applications

- **Medical imaging:** enhancing and segmenting scans.
- **Document scanning and OCR:** straightening, thresholding and cleaning pages before text recognition.
- **Quality inspection:** finding defects on a factory line.
- **Machine learning:** resizing, normalising and augmenting images before training.
- **Remote sensing and agriculture:** analysing satellite or drone images.

## 7. Conclusion

Python makes image processing approachable. Since images are arrays, a few lines of OpenCV and NumPy can already convert, filter, segment and analyse them. The example here covers the basic pipeline of loading, cleaning, enhancing, segmenting and detecting. The same steps are the foundation for more advanced work such as object detection and deep learning.

## 8. References

1. OpenCV. *OpenCV-Python Tutorials.* https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
2. Pillow. *Pillow (PIL Fork) Documentation.* https://pillow.readthedocs.io/
3. scikit-image. *scikit-image: Image processing in Python.* https://scikit-image.org/docs/stable/
4. NumPy. *NumPy Documentation.* https://numpy.org/doc/
5. Matplotlib. *Matplotlib Documentation.* https://matplotlib.org/stable/
6. OpenCV. *Image Thresholding (Otsu's binarization).* https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
7. OpenCV. *Canny Edge Detection.* https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
