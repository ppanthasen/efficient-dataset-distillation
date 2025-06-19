# Dataset Distillation for Efficient ML Training (DataDAM & PAD)

This repository implements a standalone dataset distillation pipeline using two state-of-the-art methods:  
- **DataDAM** (Dataset Distillation with Attention Matching)
- **PAD** (Prioritizing Alignment in Dataset Distillation)
  
The project was developed as an independent submission for ECE1512 at the University of Toronto.


## 🧠 Project Overview

Dataset distillation aims to generate a compact synthetic dataset that enables deep learning models to generalize as effectively as if trained on the full dataset. This implementation explores:

- **DataDAM**: Enhances dataset distillation by aligning attention maps between synthetic and real samples, ensuring the distilled data preserves how the model internally focuses on different input regions.
- **PAD**: Prioritizes important trajectories and filters redundant samples to improve efficiency


## 💡 Key Results

- Reduced dataset size by **98.6%**
- Preserved model performance with **≤ 10% drop in accuracy**
- Achieved **30% improvement** in continual learning under memory constraints
- Cut training time by up to **one-third** compared to full-data training

## 📂 Contents

- `main.ipynb` — Main notebook implementing dataset distillation with DataDAM and PAD  
- `model_train.py` / `model_test.py` — Scripts for training and evaluation  
- `networks.ipynb` / `utils.ipynb` — Provided support code from course material  
- `utils_ext.py` — Custom utility functions  
- `data/` — Dataset folder (e.g., MHIST) <!--; excluded from version control  -->
- `runs/` — Output logs/checkpoints <!--(optional; excluded from version control)  -->

## 🛠️ Run Locally

```bash
git clone https://github.com/ppanthasen/YOUR_REPO_NAME.git
cd ProjectA
pip install torch numpy matplotlib pandas scikit-learn Pillow thop tqdm
jupyter notebook projectA.ipynb
```

## 📈 Sample Results

### 🔹 Synthetic Image Quality


<p align="center">
  <img src="https://github.com/user-attachments/assets/8995aca2-3970-44b5-8fdc-69df7fcd98a3" height="300"/>
</p>


> Visualization of MNIST synthetic images generated from noise initialization. Each row shows a digit class; columns depict progression from noise to optimized samples over 100 epochs. Despite aggressive compression (e.g., 10–100 images per class), the distilled data successfully captures class structure and identity.*


### 🔹 Continual Learning Performance
<p align="center">
  <img src="https://github.com/user-attachments/assets/a86f7e8d-3186-4c28-aa92-0376017c131b" height="200"/>
</p>


> Test accuracy across varying numbers of classes using different buffer sizes and image-per-class (IPC) settings. IPC100 with buffer 1000 achieves the highest robustness. IPC10 demonstrates that highly compressed synthetic datasets can still perform competitively in continual learning scenarios.*


### 🔹 Training Time vs Accuracy

| **Dataset** | **Type**         | **Epochs** | **Test Acc (%)** | **Training Time (s)** |
|-------------|------------------|------------|------------------|------------------------|
| MNIST       | Full             | 20         | 99.0             | 193.6                  |
|             | Syn (real-init)  | 20 / 100   | 79.0 / 91.0      | 13.0 / 61.6            |
|             | Syn (noise-init) | 20 / 100   | 78.5 / 88.8      | 12.7 / 59.9            |
| MHIST       | Full             | 20         | 78.1             | 85.6                   |
|             | Syn (real-init)  | 20 / 100   | 57.1 / 66.2      | 13.8 / 67.7            |
|             | Syn (noise-init) | 20 / 100   | 43.2 / 40.7      | 12.8 / 59.9            |

> *Synthetic datasets reduce training time by up to 90%, while retaining strong accuracy, especially when initialized from real data. Results confirm the effectiveness of distilled data in fast, low-resource training environments.*


### 🔹 Generalization Across Architectures

| **Dataset** | **Model**  | **Acc (%) @ 20 Epochs** | **@ 100 Epochs** | **@ Max Epochs** | **# Max Epochs** |
|-------------|------------|--------------------------|------------------|-------------------|----------------|
| MNIST       | LeNet      | 8.92                     | 21.81            | 84.96             | 480            |
|             | AlexNet    | 11.22                    | 20.44            | 84.88             | 412            |
|             | VGG-11     | 80.93                    | 88.30            | 88.81             | 201            |
| MHIST       | LeNet      | 36.85                    | 62.74            | 60.70             | 1000           |
|             | AlexNet    | 63.15                    | 56.60            | 54.04             | 1000           |
|             | VGG-11     | 58.03                    | 58.14            | 60.90             | 1000           |

> *Distilled datasets generalize well across model architectures. VGG-11 exhibits strong baseline alignment even with fewer training epochs. Performance degradation is minimal on MNIST and reasonably stable on MHIST.*



## 📃 Acknowledgements

Some parts of this project were adapted from existing public and instructional sources:

- Code and methodology from the original **[DataDAM](https://github.com/DataDistillation/DataDAM)** and **[PAD](https://github.com/NUS-HPC-AI-Lab/PAD)** repositories  
- Course-provided utilities:
  - `networks.ipynb`
  - `utils.ipynb`

## 📘 License

This project is provided for academic and portfolio use only.

