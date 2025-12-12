# Design-GenNO: Physics-Informed Generative Neural Operator for Inverse Microstructure Design

This repository provides the official implementation of [**Design-GenNO**](https://www.sciencedirect.com/science/article/pii/S0045782525008692), a novel **physics-informed generative model based on neural operators** for **inverse microstructure design**.

---

## 🌟 Overview

**Design-GenNO** is a physics-informed deep generative neural operator framework for inverse microstructure design. It unifies **generative modeling**, **operator learning**, and **physics-informed training** to design microstructures that meet user-specified targets such as effective properties or microscopic field responses.

### ✨ Key Features

- **Generative Neural Operator framework** combining MultiONet decoders with a structured latent space
- **Physics-informed training** using PDE residuals, reducing reliance on costly labeled data
- **Normalizing Flow prior** for efficient sampling and robust optimization
- Supports diverse inverse design tasks:
  - Property-matching problems
  - Microstructure recovery from field measurements
  - Maximization of anisotropic conductivity ratios
- Demonstrates strong **out-of-distribution generalization** beyond training data

---

## 🎯 Inverse Microstructure Design Problems

This framework addresses three key inverse design problems:

### Problem 1: Property-Targeted Design

**Objective:** Given a target region $\mathcal{T}_d$ in the effective property space, design microstructures $\mu$ whose effective property $\kappa_{\text{eff}}$ lies within $\mathcal{T}_d$.

**Applications:** Materials with specified thermal/electrical conductivity, stiffness, or other bulk properties.

### Problem 2: Field Response Matching

**Objective:** Given a target field $u_d$ on domain $\Omega$, identify microstructure(s) $\mu$ such that the corresponding field prediction satisfies $u \approx u_d$.

**Applications:** Temperature distribution matching, stress field engineering, wave propagation control.

### Problem 3: Property Optimization

**Objective:** Given a utility function $F$, design microstructures $\mu$ that maximize the objective $F(\kappa_{\text{eff}})$.

**Applications:** Maximizing anisotropic conductivity ratios, optimizing thermal performance, enhancing mechanical properties.

---

## ⚙️ Dependencies

The implementation relies on the following Python packages:

```bash
torch==2.2.0
scipy==1.12.0
tqdm==4.66.1
h5py==3.10.0
matplotlib==3.8.2
scikit-learn==1.7.2
scienceplots==2.1.1
```

---

## 📂 Datasets and Trained Models

All datasets and trained models are publicly available on Kaggle: **[Design-GenNO Dataset](https://www.kaggle.com/datasets/yhzang32/design-genno)**

### 📦 Data Structure

- `Dataset/` - Training and testing datasets
- `saved_models/` - Pretrained Design-GenNO models

Download these folders and place them in the same directory as the training scripts.

---

## 🚀 Usage Instructions

### Step 1. Configure the Environment

Install the required Python packages:

```bash
pip install torch==2.2.0 scipy==1.12.0 tqdm==4.66.1 h5py==3.10.0 matplotlib==3.8.2 scikit-learn scienceplots
```

### Step 2. Prepare Data and Models

Download the datasets and pretrained models from the [Kaggle dataset](https://www.kaggle.com/datasets/yhzang32/design-genno) and place them in the project directory.

### Step 3. Train Design-GenNO Model

**Mixed-driven training** (using both labeled data and PDE residuals):

```bash
nohup python3 -u DesignGenNO_MixedDriven.py > out_mixed &
```

**Physics-driven training** (using only PDE residuals):

```bash
nohup python3 -u DesignGenNO_PhysicsDriven.py > out_physics &
```

### Step 4. Solve Inverse Design Problems

Navigate to the corresponding Jupyter Notebook for your problem of interest and execute it to reproduce the numerical results.

**Example notebooks:**

- `InverseDesign_P1_region_case1.ipynb` - Design microstructures with target effective properties (in-distribution)
- `InverseDesign_P1_region_case2.ipynb` - Design microstructures with target effective properties (out-of-distribution)
- `InverseDesign_P2_target_T.ipynb` - Microstructure recovery from field measurements
- `InverseDesign_P3_maximize.ipynb` - Maximize anisotropic conductivity ratios

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{zang2026design,
  title={Design-GenNO: A physics-informed generative model with neural operators for inverse microstructure design},
  author={Zang, Yaohua and Koutsourelakis, Phaedon-Stelios},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={450},
  pages={118597},
  year={2026},
  publisher={Elsevier}
}
```