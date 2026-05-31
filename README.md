# Lung Disease Detection from Chest X-Rays

Analyzed **5,863 pediatric chest X-ray images** to build a pneumonia classification system.  
Compared 5 approaches - custom CNN achieved **97.7% accuracy**, outperforming all pretrained models.

-----

## Results

| Model | Accuracy |
|-------|----------|
| **Custom 8-layer CNN** | **97.70%** (Best) |
| DenseNet | 87.18% |
| ResNet50 | 84.13% |
| InceptionNet | 76.76% |
| VGG16 | 66.19% |

<img width="1154" height="811" alt="image" src="https://github.com/user-attachments/assets/92e3f6e5-e9bd-42eb-bfe8-81731860848f" />


> Key finding: a custom-built CNN trained specifically on this dataset outperformed all 4 pretrained transfer learning models.

---

## The Analysis Pipeline

| Step | Notebook | What I found |
|------|----------|--------------|
| 1. Data cleaning | `1_dataBalancing.ipynb` | Dataset had class imbalance - fixed via augmentation |
| 2. Feature extraction | `2_featureExtraction.ipynb` | Edge filters captured lung structure better than raw pixels |
| 3. Segmentation | `3_lungs-segmentation.ipynb` | K-means clustering isolated lung regions from background noise |
| 4. Modeling | `4_CNN.ipynb` | Custom CNN hit 97.70% — best performer |
| 5. Comparison | `5_Transfer-learning-pneumonia-detection.ipynb` | Pretrained models ranged from 66–87% on this dataset |
| 6. Visualization | `6_Heat_maps.ipynb` | Grad-CAM confirmed model focuses on lung regions, not artifacts |
| 7. Final report | `7_compare.ipynb` | Side-by-side comparison of all 5 models |

---

## Dataset

- **5,863 pediatric chest X-ray images**
- Source: [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Binary classification: **Normal vs Pneumonia**

----

## Key Findings

- Custom CNN outperformed all pretrained models by 10+ percentage points
- Preprocessing (K-means lung segmentation + edge detection) was the biggest driver of accuracy
- Class imbalance was the main data quality issue - real-time augmentation solved it
- Grad-CAM confirmed the model focuses on lung tissue, not image artifacts or background

---

## Tech Stack

`Python` `TensorFlow` `Keras` `OpenCV` `Scikit-learn` `NumPy` `Matplotlib` `Seaborn`

---

## Setup

```bash
git clone https://github.com/tulsipatil/LungsDiseaseDetection
cd LungsDiseaseDetection
pip install tensorflow keras opencv-python scikit-learn matplotlib seaborn numpy
jupyter notebook
```

Run notebooks in order: 1 → 2 → 3 → 4 → 5 → 6 → 7

---

## Sample Output

Grad-CAM heatmaps show exactly where the model looks when making predictions.  
Red/yellow = high attention areas. The model focuses on lung tissue.
<img width="986" height="488" alt="image1" src="https://github.com/user-attachments/assets/30729d7d-4a0f-4072-b88c-6e10ebbecb3d" />


