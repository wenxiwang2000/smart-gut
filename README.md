# 🫀 SmartGut

**SmartGut** is an automated video-analysis pipeline for **detecting, tracking, and quantifying gut contraction regions of interest (ROIs)** in fruit fly microscopy videos.

The system combines **classical computer vision** with a **lightweight deep-learning classifier** to enable robust and reproducible contraction analysis from time-series imaging data.

🔗 **GitHub:** https://github.com/wenxiwang2000/smart-gut

---

# ✨ Overview

Gut contraction analysis from microscopy videos is challenging due to:

- motion drift
- unstable contrast
- noisy backgrounds
- ROI ambiguity across frames

SmartGut addresses these challenges through a **multi-stage detection and validation workflow** that integrates:

- **Multi-template matching (NCC)** for ROI localization
- **Canny edge filtering** for structural enhancement
- **Kalman filtering** for temporal stabilization
- **EfficientNet-B7 classification** for ROI validation
- Optional **vision–language verification** for additional confidence

The pipeline transforms raw microscopy videos into **trackable ROI signals and quantitative contraction metrics**.

---

# 🧠 Core Idea

SmartGut combines complementary strategies to achieve robust ROI tracking:

1. **Template-based localization**  
   Detects candidate gut regions using normalized cross-correlation.

2. **Image preprocessing**  
   Enhances contrast and structural edges for improved detection.

3. **Temporal smoothing**  
   Stabilizes ROI tracking across frames using Kalman filtering.

4. **Deep-learning validation**  
   Uses EfficientNet-B7 to distinguish valid ROIs from artifacts.

5. **Optional multimodal verification**  
   Vision–language models provide an independent validation layer.

This hybrid approach enables **stable and reproducible gut contraction tracking across long video sequences**.

---

# 🔬 Workflow

![SmartGut workflow guide](docs/SmartGut_page2.png)

**High-resolution version:**  
https://github.com/user-attachments/files/25622638/SmartGut_page2.tiff

---

# ⚙️ Core Pipeline

### 1. Frame preprocessing
Each frame undergoes several normalization steps:

- grayscale conversion  
- **CLAHE contrast normalization**  
- Gaussian denoising  

These steps improve signal clarity for downstream detection.

---

### 2. ROI localization

Candidate ROIs are detected using:

- **multi-template matching (NCC)**
- optional **Canny edge enhancement**

This allows SmartGut to identify user-defined gut regions even under variable imaging conditions.

---

### 3. Temporal stabilization

To reduce jitter and maintain smooth ROI trajectories:

- **Kalman filtering** is applied across frames.

---

### 4. Deep-learning classification

Detected ROIs are validated using a fine-tuned **EfficientNet-B7** model.

Training setup includes:

- Adam optimizer  
- cross-entropy loss  
- standard augmentation strategies  

This step helps remove false-positive detections.

---

### 5. Optional verification

For higher confidence, SmartGut can apply an additional:

- **vision–language verification step**

This independent validation layer improves robustness in difficult imaging conditions.

---

# 📊 Example Detection Results

| Raw microscopy frame | ROI detection |
|:--------------------:|:-------------:|
| ![](https://github.com/user-attachments/assets/86b49f28-59c1-457a-bba1-0bce4b2f6989) | ![](https://github.com/user-attachments/assets/ed18a457-1848-44cb-bbdc-3b8789c846cb) |

| Canony Filltering | Temporal tracking |
|:------------------:|:-----------------:|
| ![](https://github.com/user-attachments/assets/9050ba76-c532-4e08-87a5-d7dcaaba81b0) | ![](https://github.com/user-attachments/assets/d946c0bc-f145-4270-976e-1ca5db9ed4d9) |

These examples demonstrate the ability of SmartGut to detect, classify, and track gut contraction regions across frames.

---

# 📈 Performance

Benchmarking results show clear improvements in detection reliability:

- Template expansion improved accuracy from **50% → 75%**
- Vision–language verification increased confidence from **93% → 97%**

These results demonstrate the benefit of combining **classical computer vision with deep-learning validation**.

---

# 📦 Output

After analysis, SmartGut generates:

- `annotated.mp4` with bounding-box ROI tracking
- exported ROI coordinate tables
- matching confidence scores
- quantitative contraction metrics

These outputs support both **visual inspection and downstream quantitative analysis**.

---

# ⚙️ How to Use

### 1. Prepare ROI templates

Manually crop representative ROI templates in **PNG format**.

### 2. Organize template files

Place the templates in the designated template directory.

### 3. Run the pipeline

```bash
python combine_and_match.py test.mp4
