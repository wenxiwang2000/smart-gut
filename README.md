# SmartGut

**SmartGut** is an automated video-analysis pipeline for detecting, tracking, and analyzing user-defined fruit fly gut contraction regions of interest (ROI).

It combines classical computer vision methods with a lightweight deep-learning classifier to enable robust and quantitative contraction analysis from video data.

GitHub: https://github.com/wenxiwang2000/smart-gut

---

## Overview

SmartGut performs:

- ROI localization using multi-template matching (NCC)
- Edge-enhanced detection with Canny filtering
- Temporal stabilization via Kalman filtering
- ROI classification using fine-tuned EfficientNet-B7
- Optional vision–language model verification for higher confidence

The system outputs an annotated tracking video and exports ROI coordinates and matching scores for downstream quantitative analysis.

---

## Core Pipeline

1. Frame preprocessing  
   - Grayscale conversion  
   - CLAHE contrast normalization  
   - Gaussian denoising  

2. ROI localization  
   - Multi-template matching (normalized cross-correlation)  
   - Optional Canny edge enhancement  

3. Temporal smoothing  
   - Kalman filter to reduce jitter  

4. Deep-learning classification  
   - EfficientNet-B7  
   - Adam optimizer  
   - Cross-entropy loss  
   - Standard augmentation  

5. Optional verification  
   - Vision–language model for independent validation  

---

## Output

- `annotated.mp4` with bounding-box ROI tracking
- Exported ROI coordinates
- Matching confidence scores
- Quantitative contraction metrics

---

## Performance

- Template expansion improved accuracy from 50% → 75%
- Vision–language verification increased detection confidence from 93% → 97%

---

## How to Use

1. Manually crop representative ROI templates (PNG format)
2. Place templates in the designated folder
3. Run:

```bash
python combine_and_match.py test.mp4
<img width="434" height="546" alt="image" src="https://github.com/user-attachments/assets/f00fda06-c04f-4c1c-bf30-d40dc56a7f1c" />
