# CUDA NPP Image Filter Suite

This project is a CUDA-accelerated image filtering tool built using NVIDIA Performance Primitives (NPP). It supports applying multiple filters to grayscale .bmp or .pgm images efficiently on the GPU.

---

## 📌 Features

The available filters include:

### 🔍 Edge & Blur Filters

- **Canny Filter (`canny`)**
  - Detects edges using a multi-stage process: Gaussian smoothing, gradient calculation, non-maximum suppression, and hysteresis thresholding.
  - Ideal for edge detection with noise suppression.

- **Sobel Filter (`sobel`)**
  - Computes the first-order derivative in the horizontal direction.
  - Highlights vertical edges or intensity transitions.

- **Gaussian Blur (`gauss`)**
  - Applies a 3×3 Gaussian kernel to reduce image noise and detail.
  - Useful as a preprocessing step or for denoising.

- **Sharpen Filter (`sharpen`)**
  - Enhances edges by increasing contrast between adjacent pixels.
  - Makes images appear crisper and more defined.

### 📦 Smoothing & Detail Filters

- **Box Filter (`box`)**
  - Averages pixel values in a 3×3 window.
  - Simple form of image smoothing or low-pass filtering.

- **Laplacian Filter (`laplacian`)**
  - Applies a second-order derivative operation.
  - Enhances areas of rapid intensity change (edges), useful for fine detail enhancement.

---
---

## 🧱 Requirements

* CUDA Toolkit (v11+ recommended)
* NVIDIA GPU with compute capability ≥ 3.5
* g++ compiler with C++17 support
* Make, Bash

---

## 🔧 Build Instructions

1. Clone the repository and enter the project directory.
2. Run the following command:

```bash
make
```

This creates the executable `nppFilters` in the root directory.

---

## 🚀 Usage

### Command-Line

```bash
./nppFilters <input_image> <filter_type> <output_path>
```

* `input_image`: Path to a `.bmp` or `.pgm` grayscale image
* `filter_type`: One of `canny`, `sobel`, `gauss`, `sharpen`, `box`, `laplacian`
* `output_path`: Output image file or directory

**Example:**

```bash
./nppFilters ./data/wolf.bmp sobel ./output/wolf_sobel.bmp
```

If the output is a directory, the file will be named automatically.

---

## 🔀 Batch Filtering

To apply all filters to all `.bmp` images in the `data/` directory, use the provided script:

```bash
./run.sh
```

* Outputs will be saved in the `output/` directory
* Logs are appended to `output/output.log`

---

## 📂 Folder Structure

```
project/
├── Common/               # Utility headers
├── data/                 # Input images
├── output/               # Output images + log
├── src/                  # Main source code
├── run.sh                # Batch script
├── Makefile              # Build configuration
└── README.md             # Project documentation
```

---

## ⚠️ Notes

* **Image format**: Input must be 8-bit grayscale `.bmp` or `.pgm`. Color or 16-bit images will fail.
* **Emboss filter** is experimental and may require image resizing or memory tuning.
* All filters synchronize and reset the CUDA device between operations for safety.

---

## 📋 Example Log

```
Processing: ./data/wolf.bmp
Saved to: ./output/wolf_sobel.bmp
Saved to: ./output/wolf_canny.bmp
...
```


---

## 📄 License

This project is licensed under the MIT License.
