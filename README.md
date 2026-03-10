# 🫀 SmartGut

**SmartGut** is an automated video-analysis pipeline for **detecting, tracking, and quantifying gut contraction regions of interest (ROIs)** in fruit fly videos.

It combines **classical computer vision** with a **lightweight deep-learning classifier** to provide robust, reproducible, and quantitative contraction analysis from microscopy video data.

**GitHub:** [wenxiwang2000/smart-gut](https://github.com/wenxiwang2000/smart-gut)

---

## ✨ Overview

SmartGut is designed to turn raw gut-contraction videos into **trackable, analyzable ROI signals** through a multi-step detection and validation workflow.

The system integrates:

- **ROI localization** using multi-template matching (**NCC**)
- **Edge-enhanced detection** with **Canny filtering**
- **Temporal stabilization** with **Kalman filtering**
- **ROI classification** using fine-tuned **EfficientNet-B7**
- Optional **vision–language model verification** for additional confidence control

The final workflow produces both **annotated tracking videos** and **structured output files** containing ROI positions, matching confidence, and contraction-related quantitative data.

---

## 🧠 Core idea

Gut contraction analysis from microscopy videos is often affected by:

- motion drift  
- unstable contrast  
- noisy backgrounds  
- ROI ambiguity across frames  

SmartGut addresses these challenges by combining:

1. **Template-based localization** for precise ROI finding  
2. **Image preprocessing** for contrast and edge enhancement  
3. **Temporal smoothing** to reduce frame-to-frame jitter  
4. **Deep-learning classification** to validate ROI identity  
5. Optional **secondary model verification** to further improve confidence

This makes the pipeline suitable for **quantitative and reproducible contraction tracking** in user-defined gut regions.

---

## 🔬 Core pipeline

### 1. Frame preprocessing
Each video frame is prepared before ROI detection using:

- grayscale conversion  
- **CLAHE** contrast normalization  
- Gaussian denoising  

This improves signal clarity and makes template matching more stable.

### 2. ROI localization
Candidate ROIs are detected using:

- multi-template matching with **normalized cross-correlation (NCC)**
- optional **Canny edge enhancement** for sharper structural matching

This allows the system to identify user-defined gut regions even under variable image conditions.

### 3. Temporal stabilization
To reduce jitter and improve continuity across frames, ROI positions are smoothed with a:

- **Kalman filter**

This stabilizes tracking and improves downstream quantitative measurements.

### 4. Deep-learning classification
Detected ROIs are further evaluated using a fine-tuned:

- **EfficientNet-B7**

Training setup includes:

- **Adam optimizer**
- **cross-entropy loss**
- standard image augmentation

This classification step helps distinguish valid ROIs from incorrect detections.

### 5. Optional verification
For higher confidence, SmartGut can apply an optional:

- **vision–language model verification**

This serves as an additional independent validation layer for difficult or uncertain detections.

---

## 🖼️ Workflow guide

![SmartGut workflow guide](docs/SmartGut_page2.png)

**High-resolution figure:** [SmartGut_page2.tiff](https://github.com/user-attachments/files/25622638/SmartGut_page2.tiff)

---

## 📦 Output

After analysis, SmartGut generates:

- `annotated.mp4` with bounding-box ROI tracking
- exported ROI coordinates
- matching confidence scores
- quantitative contraction metrics

These outputs support both **visual inspection** and **downstream quantitative analysis**.

---

## 📈 Performance

Current benchmarking showed clear improvement across the detection workflow:

- template expansion improved accuracy from **50% → 75%**
- vision–language verification increased confidence from **93% → 97%**

These results suggest that combining classical vision with model-based verification can substantially improve ROI detection robustness.

---

## ⚙️ How to use

### 1. Prepare ROI templates
Manually crop representative ROI templates in **PNG** format.

### 2. Organize template files
Place the templates in the designated input folder.

### 3. Run the pipeline

```bash
python combine_and_match.py test.mp4
